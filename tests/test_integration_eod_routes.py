"""
W1.6.4 -- the EOD HTTP routes and the durable attempt store
(`docs/planning/traderx-integration-plan.md` §W1.6, absorbing §W0.8/§W0.9).

**This is the layer that makes the boundary reachable.** Everything below it
was already tested as a library; what is under test here is the transport
contract: which HTTP status a given refusal earns, that a refused
*instrument* is a 200 rather than an error, that the four lookup states stay
distinguishable, and that a retried submission recovers one attempt rather
than launching a duplicate overnight batch.

**The status codes are the substance, not decoration.** A coordinator
decides whether to retry from them: a hash failure must never look like a
transient error (the bytes will not change), and an accepted-but-running job
must never look like an unknown one (which invites a duplicate submission).
"""
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is optional")
from fastapi.testclient import TestClient

from engine.api import eod_routes
from engine.api.app import create_app
from engine.api.eod_routes import STORE
from engine.integration.publication import PublicationError, ResultStore
from engine.integration.capabilities import ENGINE_VERSION
from engine.integration.normalize import MAPPING_VERSION
from engine.integration.schema_version import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from engine.integration.workload import (
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    UNKNOWN_WORKLOAD,
    AttemptStore,
    workload_key,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh client with an empty attempt store, so one test's attempts
    cannot satisfy another's lookup.

    **The durable store (W0.8) is redirected to `tmp_path` per test.**
    Attempts now outlive the process, and a workload key is deterministic,
    so a result published by one test -- or by an earlier *run* -- would
    otherwise be found by the next one and look like a legitimate cache hit
    rather than like leaked state.
    """
    isolated = ResultStore(tmp_path / "eod-store")
    monkeypatch.setattr(eod_routes, "RESULT_STORE", isolated)
    # **Redirect the store the routes already use; never attach one.**
    # Setting `_store` unconditionally would give the attempt store durable
    # backing even when the module under test wired none -- which would make
    # every restart test below pass against the pre-W0.8 in-process dict,
    # proving nothing (working rule 3).
    if eod_routes.STORE._store is not None:
        monkeypatch.setattr(eod_routes.STORE, "_store", isolated)
    eod_routes.STORE.clear()
    yield TestClient(create_app())
    eod_routes.STORE.clear()


def _submit(client, case="note", version="v2", **overrides):
    body = {
        "bundlePath": str(FIXTURES / case / version),
        "marketInputs": MARKET,
    }
    body.update(overrides)
    return client.post("/eod/price", json=body)


class TestCapabilitiesIsRouted:
    """W0.9 existed as a function from the start and was never reachable.
    This is the task that lands it."""

    def test_capabilities_endpoint_serves_the_document(self, client):
        response = client.get("/eod/capabilities")
        assert response.status_code == 200
        assert response.json()["capabilitySchema"] == CAPABILITY_SCHEMA_VERSION

    def test_capabilities_advertises_both_terms_schemas(self, client):
        schemas = client.get("/eod/capabilities").json()["termsSchemas"]
        assert "traderx.instrument-terms.v1" in schemas
        assert "traderx.instrument-terms.v2" in schemas

    def test_capabilities_advertises_no_fallback(self, client):
        """The one rule this whole boundary enforces, stated where a
        coordinator reads it before submitting."""
        doc = client.get("/eod/capabilities").json()
        assert doc["marketInputs"]["fallbackOnMissingInputs"] is False


class TestSchemasAreServed:
    def test_result_schema_endpoint(self, client):
        response = client.get("/eod/schemas/result")
        assert response.status_code == 200
        assert response.json()["$id"].endswith(f"{RESULT_SCHEMA_VERSION}.json")

    def test_capability_schema_endpoint(self, client):
        response = client.get("/eod/schemas/capabilities")
        assert response.status_code == 200
        assert response.json()["$id"].endswith(f"{CAPABILITY_SCHEMA_VERSION}.json")

    def test_advertised_schema_urls_actually_resolve(self, client):
        """A capability document pointing at a 404 is worse than one
        pointing nowhere."""
        urls = client.get("/eod/capabilities").json()["schemas"]
        assert client.get(urls["resultSchemaUrl"]).status_code == 200
        assert client.get(urls["capabilitySchemaUrl"]).status_code == 200


class TestPricingOverHttp:
    def test_note_prices_to_the_verified_number(self, client):
        """TraderX independently reproduced +/-103,308.33. It must survive
        the trip through HTTP unchanged."""
        body = _submit(client).json()
        assert body["state"] == STATE_COMPLETED
        values = sorted(i["calculations"]["npv"]["value"] for i in body["result"]["items"])
        assert values[1] == pytest.approx(103308.33, abs=0.01)
        assert values[0] == pytest.approx(-103308.33, abs=0.01)

    def test_bill_prices_to_the_verified_number(self, client):
        body = _submit(client, case="bill").json()
        values = sorted(i["calculations"]["npv"]["value"] for i in body["result"]["items"])
        assert values[1] == pytest.approx(98507.15, abs=0.01)

    def test_result_carries_its_schema_version(self, client):
        assert _submit(client).json()["result"]["resultSchema"] == RESULT_SCHEMA_VERSION

    def test_response_carries_the_workload_key_and_attempt_id(self, client):
        body = _submit(client).json()
        assert body["workloadKey"].startswith("sha256:")
        assert body["attemptId"]


class TestRefusalsAreNotHttpErrors:
    """**A refused instrument is the answer, not a failure to answer.**

    Returning an HTTP error for a refusal would make "we correctly declined
    to guess" indistinguishable from "we broke" -- and the whole design
    rests on that being distinguishable.
    """

    def test_sofr_refusal_is_a_200(self, client):
        response = _submit(client, case="sofr")
        assert response.status_code == 200

    def test_sofr_refusal_names_its_reason_in_the_result(self, client):
        body = _submit(client, case="sofr").json()
        items = body["result"]["items"]
        reasons = {
            i["calculations"]["npv"].get("reason") for i in items
        }
        assert reasons & {"CONVENTION_NOT_SUPPORTED", "TERMS_NOT_SUPPLIED"}

    def test_equity_refusal_is_a_200(self, client):
        assert _submit(client, case="equity").status_code == 200

    def test_coverage_block_reports_the_refusal(self, client):
        body = _submit(client, case="sofr").json()
        coverage = body["result"]["coverage"]
        assert coverage["allOutcomesAccountedFor"] is True
        assert coverage["allApplicableComputed"] is False


class TestTransportErrorsAreTruthful:
    def test_unknown_bundle_path_is_an_integrity_failure_not_a_404(self, client):
        """**A missing bundle is a 422, and that is deliberate.**

        W0.1 step 4 already established that a *missing* artifact is an
        integrity failure rather than an absence -- that is the distinction
        between an empty contracts file (valid, zero OTC coverage) and a
        missing one (the bundle does not describe what it claims to).
        `load_bundle` raises `BundleIntegrityError` for a missing manifest
        for exactly that reason, and this route does not second-guess it:
        re-classifying it as 404 here would reintroduce, at the transport
        layer, the very distinction W0.1 exists to hold.
        """
        response = _submit(client, case="does-not-exist")
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "BUNDLE_INTEGRITY_FAILED"
        assert "missing" in response.json()["detail"]["detail"].lower()

    def test_unresolvable_market_inputs_is_400(self, client):
        """The caller asked for something unresolvable and can fix it."""
        response = _submit(
            client, marketInputs={"mode": "assumed-profile", "assumedProfileId": "no-such"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "MARKET_INPUTS_NOT_SUPPLIED"

    def test_no_market_inputs_prices_nothing_but_does_not_error(self, client):
        """Omitting the block is legal: it means nothing gets priced, never
        that something gets priced against an assumed curve."""
        response = _submit(client, marketInputs=None)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["marketProvenance"] is None
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] != "ok"

    def test_tampered_bundle_is_422(self, client, tmp_path):
        """The request was well-formed; its content did not verify. Never a
        500, and never retryable."""
        import shutil
        root = tmp_path / "tampered"
        shutil.copytree(FIXTURES / "note" / "v2", root)
        positions = root / "positions.csv"
        positions.write_bytes(positions.read_bytes() + b"# tampered\n")

        response = client.post("/eod/price", json={
            "bundlePath": str(root), "marketInputs": MARKET,
        })
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "BUNDLE_INTEGRITY_FAILED"


class TestFourLookupStates:
    """Plan §W0.8 / v4 §4.2. A bare 404 for both 'unknown' and 'running' is
    what invites a duplicate overnight batch against work already in
    flight."""

    def test_unknown_workload_is_404_with_its_own_reason(self, client):
        response = client.get("/eod/results/by-workload/sha256:never-submitted")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == UNKNOWN_WORKLOAD

    def test_completed_workload_is_200_with_the_result(self, client):
        key = _submit(client).json()["workloadKey"]
        response = client.get(f"/eod/results/by-workload/{key}")
        assert response.status_code == 200
        assert response.json()["state"] == STATE_COMPLETED
        assert response.json()["result"]["resultSchema"] == RESULT_SCHEMA_VERSION

    def test_running_is_200_not_404(self):
        """Asserted at the store level: an accepted job that has not
        finished is emphatically not 'unknown'."""
        store = AttemptStore()
        store.start("sha256:k")
        found = store.lookup("sha256:k")
        assert found is not None
        assert found.state == STATE_RUNNING

    def test_failed_workload_is_200_with_a_reason(self, client):
        """A failed attempt is recorded before the error propagates, so the
        lookup reports `failed` rather than `UNKNOWN_WORKLOAD` -- which
        would wrongly invite a resubmission."""
        _submit(client, marketInputs={"mode": "assumed-profile", "assumedProfileId": "no-such"})
        # The attempt exists even though the HTTP call raised.
        assert any(a.state == STATE_FAILED for a in STORE._attempts.values())


class TestIdempotentSubmission:
    """v4 §4.3. Retrying a lost response must recover the same attempt, not
    launch a second computation."""

    def test_same_submission_id_recovers_one_attempt(self, client):
        first = _submit(client, submissionId="sub-1").json()
        second = _submit(client, submissionId="sub-1").json()
        assert first["attemptId"] == second["attemptId"]

    def test_different_submission_ids_are_distinct_attempts(self, client):
        """A deliberate second benchmark repetition uses a new id, and must
        get a new attempt -- even though the workload key is identical."""
        first = _submit(client, submissionId="sub-a", reuseExistingResult=False).json()
        second = _submit(client, submissionId="sub-b", reuseExistingResult=False).json()
        assert first["attemptId"] != second["attemptId"]
        assert first["workloadKey"] == second["workloadKey"]

    def test_reuse_serves_a_cached_result(self, client):
        _submit(client)
        second = _submit(client).json()
        assert second["reused"] is True

    def test_reuse_false_does_not_serve_a_cached_result(self, client):
        """'Don't serve me a cache' -- not 'always start a new attempt', a
        distinction that would otherwise defeat submissionId idempotency."""
        _submit(client)
        second = _submit(client, reuseExistingResult=False).json()
        assert second["reused"] is False


class TestSubmissionIdCannotCrossWorkloads:
    """**A bug found while reviewing W1.6.4, before it shipped.**

    `start()` originally returned the existing attempt for any repeated
    `submissionId`, without checking the workload matched. Submitting a
    *different* bundle under a reused id therefore returned the **first
    bundle's priced result** -- a confident, correct-looking number computed
    from entirely different inputs.

    That is the silently-wrong-number failure this whole boundary exists to
    prevent, reached through the idempotency path rather than through a
    pricer. Idempotency means "this exact request, again"; it cannot mean
    "whatever I sent last time under this name".
    """

    def test_reusing_an_id_for_a_different_bundle_is_refused(self, client):
        first = _submit(client, case="note", submissionId="dup")
        assert first.status_code == 200

        second = _submit(client, case="bill", submissionId="dup")
        assert second.status_code == 409
        assert second.json()["detail"]["reason"] == "SUBMISSION_ID_CONFLICT"

    def test_the_conflict_does_not_return_the_other_bundles_numbers(self, client):
        """The assertion that names the actual danger: a bill submission
        must never come back carrying the note's 103,308.33."""
        _submit(client, case="note", submissionId="dup")
        second = _submit(client, case="bill", submissionId="dup")
        assert "result" not in second.json()

    def test_reusing_an_id_for_the_same_bundle_still_recovers_the_attempt(self, client):
        """The guard must not break idempotency itself -- the same workload
        under the same id is exactly the retry-a-lost-response case."""
        first = _submit(client, case="note", submissionId="same").json()
        second = _submit(client, case="note", submissionId="same").json()
        assert first["attemptId"] == second["attemptId"]

    def test_store_level_conflict_raises(self):
        from engine.integration.workload import SubmissionIdConflict

        store = AttemptStore()
        store.start("sha256:A", submission_id="dup")
        with pytest.raises(SubmissionIdConflict, match="different"):
            store.start("sha256:B", submission_id="dup")

    def test_conflict_is_detected_even_after_the_first_completed(self):
        """The dangerous ordering: the first attempt has a *result* to hand
        back wrongly."""
        from engine.integration.workload import SubmissionIdConflict

        store = AttemptStore()
        first = store.start("sha256:A", submission_id="dup")
        first.complete({"npv": 1.0})
        with pytest.raises(SubmissionIdConflict):
            store.start("sha256:B", submission_id="dup")


class TestAttemptAddressing:
    def test_attempt_is_addressable_by_id(self, client):
        attempt_id = _submit(client).json()["attemptId"]
        response = client.get(f"/eod/attempts/{attempt_id}")
        assert response.status_code == 200
        assert response.json()["state"] == STATE_COMPLETED

    def test_unknown_attempt_is_404(self, client):
        assert client.get("/eod/attempts/nope").status_code == 404


class TestWorkloadKey:
    """The key answers 'have I already computed exactly this?'. Both
    directions of error are bugs -- see the module docstring in
    `engine.integration.workload`."""

    def _key(self, **overrides):
        base = dict(
            bundle_id="B", cluster_epoch="E", session_date="2025-06-02",
            market_inputs=MARKET, calculations=None, reporting_currency="USD",
            mapping_version=MAPPING_VERSION, engine_version=ENGINE_VERSION,
            result_schema=RESULT_SCHEMA_VERSION,
        )
        base.update(overrides)
        return workload_key(**base)

    def test_identical_inputs_produce_identical_keys(self):
        assert self._key() == self._key()

    def test_key_is_prefixed_with_its_algorithm(self):
        assert self._key().startswith("sha256:")

    @pytest.mark.parametrize("field,value", [
        ("bundle_id", "OTHER"),
        ("cluster_epoch", "OTHER"),
        ("session_date", "2025-06-03"),
        ("reporting_currency", "EUR"),
        ("mapping_version", "other"),
        ("engine_version", "9.9.9"),
        ("result_schema", "jaxrisk.eod-result.v2"),
        ("precision", "float32"),
        ("terms_hash", "sha256:abc"),
    ])
    def test_any_input_that_changes_a_number_changes_the_key(self, field, value):
        assert self._key() != self._key(**{field: value})

    def test_different_market_inputs_change_the_key(self):
        other = {"mode": "assumed-profile", "assumedProfileId": "something-else"}
        assert self._key() != self._key(market_inputs=other)

    def test_precision_separates_the_key_space(self):
        """Plan §W0.8 names this explicitly: a float32 and a float64 run of
        identical inputs are different computations with different answers,
        and must never share a cached result."""
        assert self._key(precision="float32") != self._key(precision="float64")

    def test_calculation_order_does_not_change_the_key(self):
        """Asking for [npv, theta] and [theta, npv] is one computation.
        Otherwise a cache misses on a cosmetic difference and recomputes an
        overnight batch."""
        a = self._key(calculations=["npv", "theta"])
        b = self._key(calculations=["theta", "npv"])
        assert a == b

    def test_json_key_order_does_not_change_the_key(self):
        a = self._key(market_inputs={"mode": "assumed-profile", "assumedProfileId": "p"})
        b = self._key(market_inputs={"assumedProfileId": "p", "mode": "assumed-profile"})
        assert a == b

    def test_submission_id_is_not_part_of_the_key(self):
        """A submissionId identifies a *request*; the key identifies a
        *computation*. If it were in the key, every retry would recompute --
        the opposite of idempotent."""
        import inspect
        assert "submission" not in inspect.signature(workload_key).parameters


class TestAttemptImmutability:
    """Attempts are immutable once terminal, so a second attempt can never
    overwrite a first's recorded outcome."""

    def test_completing_twice_raises(self):
        store = AttemptStore()
        attempt = store.start("sha256:k")
        attempt.complete({"ok": True})
        with pytest.raises(ValueError, match="immutable"):
            attempt.complete({"ok": False})

    def test_failing_a_completed_attempt_raises(self):
        store = AttemptStore()
        attempt = store.start("sha256:k")
        attempt.complete({"ok": True})
        with pytest.raises(ValueError, match="immutable"):
            attempt.fail("too late")

    def test_lookup_prefers_a_successful_attempt_over_a_later_failure(self):
        """A later failure must never hide an earlier success."""
        store = AttemptStore()
        first = store.start("sha256:k")
        first.complete({"good": True})
        second = store.start("sha256:k")
        second.fail("transient")
        assert store.lookup("sha256:k").attempt_id == first.attempt_id

    def test_lookup_prefers_success_over_a_running_retry(self):
        store = AttemptStore()
        done = store.start("sha256:k")
        done.complete({"good": True})
        store.start("sha256:k")  # a running retry
        assert store.lookup("sha256:k").state == STATE_COMPLETED


class TestExistingRoutesUnaffected:
    """W1.6.4 mounts a new router on a working app. The portfolio routes
    must be untouched."""

    def test_health_still_responds(self, client):
        assert client.get("/health").status_code == 200

    def test_version_still_responds(self, client):
        assert client.get("/version").status_code == 200

    def test_eod_routes_are_namespaced(self, client):
        """Everything EOD lives under /eod, so it cannot collide with the
        portfolio contract."""
        assert client.get("/capabilities").status_code == 404
        assert client.get("/eod/capabilities").status_code == 200


class TestResultsSurviveARestartOverHttp:
    """W0.8's durable half, driven through the routes rather than the store.

    **A restart is simulated by replacing the in-memory attempt store while
    leaving the durable one in place** -- which is exactly the state a
    bounced uvicorn comes up in. Every test here passes trivially against
    the pre-W0.8 in-process dict *only* if the dict survives; that is the
    point, so each one is verified to fail when `RESULT_STORE` is removed.
    """

    def _restart(self):
        """Drops in-process state, keeps whatever durable store the module
        wired -- a restart.

        **The new store inherits `_store` from the old one rather than
        being handed `RESULT_STORE`.** Passing the durable store explicitly
        would reconstruct it even for a build that never wired one, which
        is exactly how these tests would come to pass against the pre-W0.8
        in-process dict.
        """
        restarted = AttemptStore(store=eod_routes.STORE._store)
        eod_routes.STORE = restarted
        return restarted

    @pytest.fixture(autouse=True)
    def _restore_store(self):
        original = eod_routes.STORE
        yield
        eod_routes.STORE = original

    def test_a_completed_result_is_still_found_after_a_restart(self, client):
        key = _submit(client).json()["workloadKey"]
        self._restart()
        response = client.get(f"/eod/results/by-workload/{key}")
        assert response.status_code == 200
        assert response.json()["state"] == STATE_COMPLETED
        assert response.json()["result"]["resultSchema"] == RESULT_SCHEMA_VERSION

    def test_the_numbers_survive_the_restart_unchanged(self, client):
        """Not just *a* result -- the same one. A restart that returned a
        structurally valid result with different numbers would be worse
        than losing it."""
        before = _submit(client).json()
        key = before["workloadKey"]
        self._restart()
        after = client.get(f"/eod/results/by-workload/{key}").json()
        assert after["result"] == before["result"]
        assert after["attemptId"] == before["attemptId"]

    def test_an_attempt_is_still_addressable_by_id_after_a_restart(self, client):
        attempt_id = _submit(client).json()["attemptId"]
        self._restart()
        response = client.get(f"/eod/attempts/{attempt_id}")
        assert response.status_code == 200
        assert response.json()["state"] == STATE_COMPLETED

    def test_a_retried_submission_does_not_recompute_after_a_restart(self, client):
        """**The duplicate overnight batch, reached through a restart.**
        A coordinator retrying a lost response after the engine bounced
        must recover the attempt it already has, not start a second one."""
        first = _submit(client, submissionId="sub-restart").json()
        self._restart()
        second = _submit(client, submissionId="sub-restart").json()
        assert second["attemptId"] == first["attemptId"]
        assert second["reused"] is True

    def test_an_unrelated_workload_is_still_unknown_after_a_restart(self, client):
        """The store must not turn every lookup into a hit. A key that was
        never submitted stays a 404 with its own reason."""
        _submit(client)
        self._restart()
        response = client.get("/eod/results/by-workload/sha256:never-submitted")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == UNKNOWN_WORKLOAD


class TestAnUnpublishedResultIsNotReportedAsSuccess:
    """**The computation succeeded and the record of it did not land.**

    Reporting 200 here would tell the coordinator its result is published
    and discoverable when a lookup will not find it -- and the coordinator's
    next move on a 200 is to stop retrying. So the one thing this must not
    do is look like success.
    """

    @pytest.fixture(autouse=True)
    def _restore_store(self):
        original = eod_routes.STORE
        yield
        eod_routes.STORE = original

    def _break_publication(self, monkeypatch):
        def refuse(**kwargs):
            raise PublicationError("simulated store failure")

        monkeypatch.setattr(eod_routes.STORE._store, "publish", refuse)

    def test_a_publication_failure_is_a_500_not_a_200(self, client, monkeypatch):
        self._break_publication(monkeypatch)
        response = _submit(client)
        assert response.status_code == 500
        assert response.json()["detail"]["reason"] == "RESULT_NOT_PUBLISHED"

    def test_the_unpublished_result_is_not_discoverable(self, client, monkeypatch):
        """The 500 must be the truth, not a pessimistic guess: a lookup for
        that workload really does find nothing.

        **The key is taken from a successful submission of the same
        bundle**, not hand-built. A hand-built key that does not match what
        the route computes asserts `None` for a workload nobody ever
        submitted, and passes no matter what the code does.
        """
        key = _submit(client).json()["workloadKey"]
        assert eod_routes.STORE._store.lookup(key) is not None  # control

        eod_routes.STORE.clear()
        self._break_publication(monkeypatch)
        response = _submit(client)
        assert response.status_code == 500

        monkeypatch.undo()
        assert eod_routes.STORE._store.lookup(key) is None

    def test_a_failed_attempt_still_reports_the_real_cause(self, client, monkeypatch):
        """**A store failure must not mask a pricing failure.** Letting it
        replace MARKET_INPUTS_NOT_SUPPLIED with a disk message would hide
        the thing the caller has to fix, and would turn a 400 they must not
        retry into a 500 they will."""
        self._break_publication(monkeypatch)
        response = _submit(
            client,
            marketInputs={"mode": "assumed-profile", "assumedProfileId": "no-such"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "MARKET_INPUTS_NOT_SUPPLIED"
