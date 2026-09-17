"""
W0.8 -- crash-safe publication and the durable result store
(`docs/planning/traderx-integration-plan.md` §W0.8, closing the half that
W1.6.4 left open; `docs/known-issues.md` I-08).

**What is under test here is what survives a crash, not what happens when
nothing goes wrong.** The state machine and the workload key were already
covered by `test_integration_eod_routes.py`; every one of those tests passes
against a pure in-memory store. These tests are the ones that fail against
it.

The plan names four crash windows and this file drives each of them
directly, by doing to the store exactly what a crash at that instant would
leave behind:

  1. kill between artifact write and manifest publish -> lookup finds
     nothing, and in particular finds no *partial* result
  2. kill after publish -> lookup finds the complete result
  3. **kill between publish and pointer advance -> lookup still finds it**,
     via the scan. This is the window TraderX found in v3 and the reason
     the manifest, not the pointer, is the commit point.
  4. restart -> completed attempts survive; running ones do not, and are
     reported honestly as unknown rather than as something else

**Why simulate crashes by manipulating the store rather than killing a
process.** A real `SIGKILL` mid-publication is not reproducible on demand:
you cannot land it in the one-instruction window between the rename and
the pointer write. Removing the pointer *after* a successful publication
produces the identical on-disk state, deterministically, and that state is
the thing the recovery path actually has to cope with. The alternative --
a test that kills a subprocess and hopes -- would be flaky in exactly the
direction that trains people to ignore it (working rule 10).
"""
import json
from pathlib import Path

import pytest

from engine.integration.publication import (
    PUBLICATION_SCHEMA,
    PublicationError,
    ResultStore,
    default_store_root,
)
from engine.integration.workload import (
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    AttemptStore,
)

KEY = "sha256:" + "a" * 64
OTHER_KEY = "sha256:" + "b" * 64
RESULT = {"resultSchema": "jax.eod-result.v1", "bundleId": "B1", "items": []}


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "store")


def _publish(store, attempt_id, key=KEY, state=STATE_COMPLETED, **kwargs):
    return store.publish(
        attempt_id=attempt_id,
        workload_key=key,
        state=state,
        result=RESULT if state == STATE_COMPLETED else None,
        **kwargs,
    )


class TestTheStoreLayoutIsWhatTheProtocolNeeds:
    """The four-step protocol's guarantees rest on filesystem behaviour
    that only holds under specific conditions. These pin the conditions."""

    def test_temp_directory_is_a_sibling_of_attempts(self, store):
        """`os.replace` is atomic only *within* a filesystem. A temp dir on
        another device turns step 3 from a rename into a copy -- or into a
        `OSError`, on a machine that may not be the developer's."""
        assert (store.root / "tmp").parent == (store.root / "attempts").parent

    def test_publication_leaves_no_temp_files_behind(self, store):
        _publish(store, "att-1")
        assert list((store.root / "tmp").iterdir()) == []

    def test_pointer_filename_contains_no_colon(self, store):
        """A workload key is `sha256:<hex>`, and a colon opens an alternate
        data stream on Windows. The key is digested, not escaped."""
        _publish(store, "att-1")
        names = [p.name for p in (store.root / "pointers").iterdir()]
        assert names and all(":" not in name for name in names)

    def test_manifest_records_its_own_schema(self, store):
        """So a future reader can refuse an unfamiliar layout rather than
        guess at it -- the same reason the bundle loader pins its schema."""
        manifest = _publish(store, "att-1")
        assert manifest["publicationSchema"] == PUBLICATION_SCHEMA


class TestCrashBeforeManifestPublish:
    """Window 1: killed between writing artifacts and publishing the
    manifest. **Nothing partial may be discoverable.**"""

    def test_staged_bytes_are_not_discoverable(self, store):
        """A manifest sitting in `tmp/` is not a published result, however
        complete its contents happen to be."""
        staged = store.root / "tmp" / "att-1.deadbeef.json"
        staged.write_bytes(
            json.dumps(
                {
                    "publicationSchema": PUBLICATION_SCHEMA,
                    "attemptId": "att-1",
                    "workloadKey": KEY,
                    "state": STATE_COMPLETED,
                    "result": RESULT,
                }
            ).encode("utf-8")
        )
        assert store.lookup(KEY) is None
        assert store.read_attempt("att-1") is None

    def test_a_truncated_manifest_is_not_served(self, store):
        """The bytes under `attempts/` are the commit point, so a torn one
        must read as absent rather than as a result with missing fields.
        Returning a partially-parsed document would be the file-level
        version of the silently-wrong-number failure."""
        _publish(store, "att-1")
        path = store.root / "attempts" / "att-1.json"
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        assert store.read_attempt("att-1") is None
        assert store.lookup(KEY) is None

    def test_a_foreign_schema_is_not_served(self, store):
        _publish(store, "att-1")
        path = store.root / "attempts" / "att-1.json"
        manifest = json.loads(path.read_bytes())
        manifest["publicationSchema"] = "jax.eod-publication.v99"
        path.write_bytes(json.dumps(manifest).encode("utf-8"))
        assert store.read_attempt("att-1") is None

    def test_one_unreadable_manifest_does_not_break_every_lookup(self, store):
        """An unreadable file is skipped, not raised on. One bad artifact
        from a future writer must not take out the whole store."""
        _publish(store, "att-good")
        (store.root / "attempts" / "att-bad.json").write_bytes(b"{not json")
        found = store.lookup(KEY)
        assert found is not None and found["attemptId"] == "att-good"


class TestCrashAfterPublish:
    """Window 2: the ordinary complete case."""

    def test_lookup_finds_the_complete_result(self, store):
        _publish(store, "att-1")
        found = store.lookup(KEY)
        assert found is not None
        assert found["state"] == STATE_COMPLETED
        assert found["result"] == RESULT

    def test_attempt_is_addressable_by_id(self, store):
        _publish(store, "att-1")
        assert store.read_attempt("att-1")["workloadKey"] == KEY


class TestCrashBetweenPublishAndPointerAdvance:
    """Window 3 -- **the gap TraderX found in v3.**

    A crash here leaves a complete, correct, published result whose pointer
    never advanced. A lookup that trusted the pointer would answer
    `UNKNOWN_WORKLOAD` for finished work, which is precisely what invites a
    coordinator to resubmit an overnight batch it already has the answer
    to. Every test in this class fails against a pointer-only lookup.
    """

    def _drop_pointer(self, store):
        for pointer in (store.root / "pointers").iterdir():
            pointer.unlink()

    def test_lookup_still_finds_the_result_without_a_pointer(self, store):
        _publish(store, "att-1")
        self._drop_pointer(store)
        found = store.lookup(KEY)
        assert found is not None
        assert found["attemptId"] == "att-1"

    def test_the_scan_advances_the_pointer_as_a_side_effect(self, store):
        """So the scan cost is paid once per crash, not on every lookup
        forever after."""
        _publish(store, "att-1")
        self._drop_pointer(store)
        assert list((store.root / "pointers").iterdir()) == []
        store.lookup(KEY)
        assert len(list((store.root / "pointers").iterdir())) == 1

    def test_a_stale_pointer_does_not_hide_a_newer_result(self, store):
        """The pointer is a cache, and this is the crash that makes it
        stale: the *second* publication commits its manifest (step 3) and
        dies before advancing the pointer (step 4). The pointer still names
        a perfectly valid older attempt, so nothing about it looks wrong --
        which is exactly why the scan has to be the authority."""
        _publish(store, "att-old")
        pointer_after_first = store._pointer_path(KEY).read_bytes()
        _publish(store, "att-new")
        store._pointer_path(KEY).write_bytes(pointer_after_first)
        assert store.lookup(KEY)["attemptId"] == "att-new"

    def test_the_pointer_never_moves_backwards(self, store):
        """A reconciling scan or a late publication must not point the
        cache at an attempt an earlier write already superseded -- every
        later lookup would then serve the older result without scanning to
        notice."""
        _publish(store, "att-old")
        _publish(store, "att-new")
        store._advance_pointer(KEY, "att-old", sequence=0)
        assert store.lookup(KEY)["attemptId"] == "att-new"

    def test_a_pointer_naming_a_nonexistent_attempt_falls_back_to_scan(self, store):
        _publish(store, "att-1")
        pointer = next((store.root / "pointers").iterdir())
        pointer.write_bytes(b"att-vanished")
        assert store.lookup(KEY)["attemptId"] == "att-1"

    def test_a_pointer_naming_another_workloads_attempt_is_refused(self, store):
        """The one case where trusting the pointer would return a result
        computed from **different inputs** -- the failure this whole
        boundary exists to prevent, reached through a corrupted cache."""
        _publish(store, "att-other", key=OTHER_KEY)
        _publish(store, "att-mine", key=KEY)
        pointer = store._pointer_path(KEY)
        pointer.write_bytes(b"att-other")
        found = store.lookup(KEY)
        assert found is not None
        assert found["attemptId"] == "att-mine"
        assert found["workloadKey"] == KEY

    def test_a_torn_pointer_falls_back_to_scan(self, store):
        _publish(store, "att-1")
        next((store.root / "pointers").iterdir()).write_bytes(b"")
        assert store.lookup(KEY)["attemptId"] == "att-1"


class TestOnlySuccessfulAttemptsAreServed:
    """Plan §W0.8: lookup returns the most recent *successful* attempt --
    never a failed, partial or in-flight one."""

    def test_a_failed_attempt_is_published_but_not_served_by_workload(self, store):
        _publish(store, "att-failed", state=STATE_FAILED, reason="boom")
        assert store.read_attempt("att-failed")["state"] == STATE_FAILED
        assert store.lookup(KEY) is None

    def test_a_later_failure_does_not_hide_an_earlier_success(self, store):
        _publish(store, "att-ok")
        _publish(store, "att-failed", state=STATE_FAILED, reason="boom")
        assert store.lookup(KEY)["attemptId"] == "att-ok"

    def test_a_failed_attempt_never_advances_the_pointer(self, store):
        _publish(store, "att-failed", state=STATE_FAILED, reason="boom")
        assert list((store.root / "pointers").iterdir()) == []

    def test_publishing_a_running_attempt_is_refused(self, store):
        """An in-flight computation has no result to commit, and publishing
        one would make it discoverable as a finished answer."""
        with pytest.raises(PublicationError, match="only terminal"):
            store.publish(
                attempt_id="att-1", workload_key=KEY, state=STATE_RUNNING
            )

    def test_a_refused_publication_writes_nothing(self, store):
        with pytest.raises(PublicationError):
            store.publish(attempt_id="att-1", workload_key=KEY, state=STATE_RUNNING)
        assert list((store.root / "attempts").iterdir()) == []


class TestStep2VerifiesWhatWasActuallyWritten:
    """Step 2 of the protocol. Hashing what was *meant* to be written
    proves nothing; a short write nobody reads back is a durable artifact
    that verifies against nothing."""

    def test_a_corrupted_write_is_refused_and_publishes_nothing(self, store, monkeypatch):
        """Simulates the filesystem returning different bytes than were
        written -- the case step 2 exists for."""
        real_read_bytes = Path.read_bytes

        def corrupt(self):
            data = real_read_bytes(self)
            if self.parent.name == "tmp":
                return data + b" "
            return data

        monkeypatch.setattr(Path, "read_bytes", corrupt)
        with pytest.raises(PublicationError, match="failed verification"):
            _publish(store, "att-1")

        monkeypatch.undo()
        assert store.read_attempt("att-1") is None
        assert store.lookup(KEY) is None
        assert list((store.root / "tmp").iterdir()) == []

    def test_the_error_names_both_digests(self, store, monkeypatch):
        """So a reader can tell a truncation from a substitution rather
        than being told only that something was wrong."""
        real_read_bytes = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: real_read_bytes(self) + (b" " if self.parent.name == "tmp" else b""),
        )
        with pytest.raises(PublicationError) as excinfo:
            _publish(store, "att-1")
        message = str(excinfo.value)
        assert "expected" in message and "actual" in message


class TestCanonicalSerialization:
    def test_key_order_does_not_change_the_published_bytes(self, store):
        """Two manifests differing only in dict order must hash the same,
        or re-verifying a published artifact would depend on how the writer
        happened to order a dict."""
        _publish(store, "att-1", submission_id="s1")
        first = (store.root / "attempts" / "att-1.json").read_bytes()
        store.clear()
        _publish(store, "att-1", submission_id="s1")
        assert (store.root / "attempts" / "att-1.json").read_bytes() == first


class TestRestartSurvival:
    """I-08's actual subject. A *new* `AttemptStore` over the *same* root is
    exactly what a restart produces."""

    def test_a_completed_attempt_survives_a_restart(self, store):
        live = AttemptStore(store=store)
        attempt = live.start(KEY, submission_id="sub-1")
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        found = restarted.lookup(KEY)
        assert found is not None
        assert found.state == STATE_COMPLETED
        assert found.result == RESULT

    def test_an_attempt_is_addressable_by_id_after_a_restart(self, store):
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        assert restarted.get(attempt.attempt_id).state == STATE_COMPLETED

    def test_idempotent_submission_survives_a_restart(self, store):
        """**The point of `submissionId`, made durable.** Without this, a
        coordinator retrying a lost response after a bounce starts a second
        computation for work already finished -- the duplicate overnight
        batch, reached through a restart instead of through a race."""
        live = AttemptStore(store=store)
        first = live.start(KEY, submission_id="sub-1")
        first.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        second = restarted.start(KEY, submission_id="sub-1")
        assert second.attempt_id == first.attempt_id
        assert second.state == STATE_COMPLETED

    def test_a_failed_attempt_survives_a_restart(self, store):
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.fail("MARKET_INPUTS_NOT_SUPPLIED: no such profile")

        restarted = AttemptStore(store=ResultStore(store.root))
        recovered = restarted.get(attempt.attempt_id)
        assert recovered.state == STATE_FAILED
        assert "MARKET_INPUTS_NOT_SUPPLIED" in recovered.reason

    def test_a_running_attempt_does_not_survive_and_says_so(self, store):
        """**Deliberate.** A running attempt is never published, because
        writing one would make an in-flight computation discoverable as a
        finished answer. Reporting it as unknown after a restart is honest;
        the coordinator resubmits and the workload key makes it the same
        computation."""
        live = AttemptStore(store=store)
        live.start(KEY)

        restarted = AttemptStore(store=ResultStore(store.root))
        assert restarted.lookup(KEY) is None

    def test_a_rehydrated_attempt_cannot_be_completed_again(self, store):
        """Immutability has to survive the restart too, or a second run
        could overwrite a first's recorded outcome through the recovery
        path."""
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        recovered = restarted.get(attempt.attempt_id)
        with pytest.raises(ValueError, match="immutable"):
            recovered.complete({"resultSchema": "different"})

    def test_a_result_published_after_a_restart_is_the_most_recent_one(self, store):
        """**Publication order has to survive the process that assigned
        it.** A sequence counter held in memory restarts at zero, so the
        *second* run's attempt claims to predate the first run's -- and a
        lookup then serves the older result while reporting it as the most
        recent successful attempt. Nothing about that output looks wrong:
        it is a real, complete, correctly-priced result for the right
        workload key, just not the current one.

        This is the test that distinguishes a disk-recovered sequence from
        a process-local counter; every other test in this file passes
        against both.
        """
        _publish(store, "att-zzz-first")

        # A restart: a brand-new store object over the same root, with no
        # memory of what the previous process had issued.
        restarted_store = ResultStore(store.root)
        restarted_store.publish(
            attempt_id="att-aaa-second",
            workload_key=KEY,
            state=STATE_COMPLETED,
            result={"resultSchema": "jax.eod-result.v1", "bundleId": "NEW", "items": []},
        )

        # **The attempt ids are chosen so the tie-break points the wrong
        # way.** With a process-local counter both attempts are sequence 0,
        # and `_scan_for_workload`'s deterministic tie-break by attempt id
        # then prefers `att-zzz-first`. Real attempt ids are uuid4, so this
        # bug would surface as a result that is stale roughly half the time
        # -- which is why the ids here are pinned rather than generated.
        assert restarted_store.lookup(KEY)["attemptId"] == "att-aaa-second"
        assert restarted_store.lookup(KEY)["result"]["bundleId"] == "NEW"

        # And a *third* reader, with no memory at all, must agree.
        assert ResultStore(store.root).lookup(KEY)["attemptId"] == "att-aaa-second"

    def test_the_sequence_high_water_mark_is_recovered_when_lost(self, store):
        """The high-water file is a cache too. Deleting it must cost a
        scan, not correctness -- otherwise a restored backup or a crash
        between the manifest rename and the high-water write would silently
        restart numbering and invert the order of everything after it."""
        live = AttemptStore(store=store)
        live.start(KEY).complete(
            {"resultSchema": "jax.eod-result.v1", "bundleId": "OLD", "items": []}
        )
        (store.root / "sequence").unlink()

        restarted_store = ResultStore(store.root)
        new = AttemptStore(store=restarted_store).start(KEY)
        new.complete({"resultSchema": "jax.eod-result.v1", "bundleId": "NEW", "items": []})

        assert ResultStore(store.root).lookup(KEY)["result"]["bundleId"] == "NEW"

    def test_a_restart_across_workloads_keeps_them_separate(self, store):
        live = AttemptStore(store=store)
        a = live.start(KEY)
        a.complete(RESULT)
        b = live.start(OTHER_KEY)
        b.complete({"resultSchema": "jax.eod-result.v1", "bundleId": "B2", "items": []})

        restarted = AttemptStore(store=ResultStore(store.root))
        assert restarted.lookup(KEY).result["bundleId"] == "B1"
        assert restarted.lookup(OTHER_KEY).result["bundleId"] == "B2"


class TestMemoryAndStorePrecedence:
    """Memory is authoritative for running state; the store is
    authoritative across restarts. Getting the order wrong inverts the
    documented precedence."""

    def test_a_running_retry_does_not_mask_a_published_success(self, store):
        live = AttemptStore(store=store)
        done = live.start(KEY, submission_id="s1")
        done.complete(RESULT)
        live.start(KEY, submission_id="s2")  # a second, still running
        assert live.lookup(KEY).attempt_id == done.attempt_id

    def test_a_published_success_outranks_an_in_memory_failure(self, store):
        """'A later failure never hides an earlier success' has to hold
        when the success is on disk and the failure is in memory."""
        first = AttemptStore(store=store)
        ok = first.start(KEY)
        ok.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        bad = restarted.start(KEY)
        bad.fail("boom")
        assert restarted.lookup(KEY).attempt_id == ok.attempt_id

    def test_memory_answers_before_the_store_is_consulted(self, store):
        """Callers hold `Attempt` objects, so a memory hit must return the
        *same object* rather than an equal copy read off disk.

        **Scope, stated honestly.** `_rehydrate`'s guard against replacing
        a live attempt is defensive: with the current call paths it cannot
        be reached, because every route into it (`get`, `lookup`,
        `start`) checks memory first and returns before touching the store.
        This test pins that precedence, which is the reachable part. It
        does **not** prove the guard itself -- a rehydration that clobbers
        passes this file, and the guard stays because it is cheap and
        because a future caller that scans first would need it. Recording
        the limit rather than implying coverage that is not there
        (working rule 9).
        """
        live = AttemptStore(store=store)
        attempt = live.start(KEY, submission_id="sub-live")
        attempt.complete(RESULT)

        assert live.get(attempt.attempt_id) is attempt
        assert live.lookup(KEY) is attempt
        assert live.start(KEY, submission_id="sub-live") is attempt

    def test_a_rehydrated_attempt_does_not_republish(self, store):
        """An attempt read back from disk must not be wired to the store.
        If it were, a terminal transition on it would rewrite a manifest
        that is supposed to be immutable."""
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        recovered = restarted.get(attempt.attempt_id)
        assert recovered._store is None


class TestMemoryOnlyRemainsTheDefault:
    """The store is optional. An `AttemptStore()` with no argument must
    behave exactly as it did before W0.8's second half, or every existing
    caller changes behaviour silently."""

    def test_no_store_means_no_publication(self, tmp_path):
        live = AttemptStore()
        attempt = live.start(KEY)
        attempt.complete(RESULT)
        assert live.lookup(KEY).state == STATE_COMPLETED

    def test_no_store_means_nothing_survives(self):
        live = AttemptStore()
        live.start(KEY).complete(RESULT)
        assert AttemptStore().lookup(KEY) is None


class TestDefaultStoreRoot:
    def test_environment_variable_selects_the_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JAX_EOD_STORE_ROOT", str(tmp_path / "configured"))
        assert default_store_root() == tmp_path / "configured"

    def test_default_is_not_the_working_directory(self, monkeypatch):
        """A store rooted at the cwd would scatter results wherever the
        service happened to be started from, making 'did this restart see
        the same store?' depend on how it was launched."""
        monkeypatch.delenv("JAX_EOD_STORE_ROOT", raising=False)
        assert default_store_root() != Path.cwd()


class TestConcurrentPublication:
    """The store is shared across FastAPI's thread pool, so read-then-
    increment of the sequence is a race unless it is locked."""

    def test_concurrent_publications_get_distinct_sequences(self, store):
        """**Without a lock, concurrent publishes read the same high-water
        mark and issue the same number.** Measured at 30 publications
        producing *2* distinct sequences before this was guarded -- which
        collapses "most recent successful attempt" into a tie-break on
        attempt id, and makes which result a lookup serves effectively
        arbitrary.
        """
        import threading

        def publish(n):
            store.publish(
                attempt_id=f"att-{n:03d}",
                workload_key=KEY,
                state=STATE_COMPLETED,
                result={"n": n},
            )

        threads = [threading.Thread(target=publish, args=(i,)) for i in range(30)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        sequences = [m["publicationSequence"] for m in store.iter_manifests()]
        assert len(sequences) == 30
        assert len(set(sequences)) == 30, "concurrent publications collided"

    def test_a_publication_and_a_lookup_do_not_deadlock(self, store):
        """`lookup` holds `_lock` and `publish` takes `_sequence_lock`.
        Two locks is two chances to order them wrongly, so this pins that
        they never wait on each other."""
        import threading

        errors = []

        def churn(n):
            try:
                store.publish(
                    attempt_id=f"att-{n:03d}",
                    workload_key=KEY,
                    state=STATE_COMPLETED,
                    result={"n": n},
                )
                assert store.lookup(KEY) is not None
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "deadlock"
        assert errors == []


class TestStep3FailureIsAPublicationError:
    """A rename that fails means the result was not published. That has to
    arrive as `PublicationError`, because the route's honest
    `RESULT_NOT_PUBLISHED` response hangs off that type -- a bare `OSError`
    escapes it and surfaces as an opaque 500 saying "something broke"
    rather than "your result is not discoverable, submit again"."""

    def test_a_failed_rename_raises_publication_error(self, store, monkeypatch):
        def refuse(src, dst):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr("engine.integration.publication.os.replace", refuse)
        with pytest.raises(PublicationError, match="could not be published"):
            _publish(store, "att-1")

    def test_a_failed_rename_leaves_nothing_behind(self, store, monkeypatch):
        def refuse(src, dst):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("engine.integration.publication.os.replace", refuse)
        with pytest.raises(PublicationError):
            _publish(store, "att-1")
        monkeypatch.undo()

        assert store.read_attempt("att-1") is None
        assert store.lookup(KEY) is None
        assert list((store.root / "tmp").iterdir()) == []


class TestAFailedPublicationLeavesNoTerminalAttemptInMemory:
    """**The manifest is the commit point -- including for the in-memory
    attempt.**

    `complete()` used to set `state = completed` and *then* publish. When
    publication raised, the attempt was left terminal in memory with nothing
    on disk: an in-process lookup reported `completed` for a result no
    restart could ever find, and the immutability guard then refused the
    retry that would have fixed it (`already completed`). The HTTP route
    returns `500 RESULT_NOT_PUBLISHED`, so the coordinator does learn to
    retry -- but the retry then hit a process whose own memory contradicted
    its disk.

    Publishing *before* the state transition makes the two agree: either the
    manifest landed and the attempt is terminal, or neither happened and the
    attempt is still running and still retryable. This is the same ordering
    rule the store itself follows -- commit first, update the cache second.
    """

    class _Boom:
        """A store that fails at step 3, after the work is done."""

        root = None

        def publish(self, **kwargs):
            raise PublicationError("disk full")

        def lookup(self, workload_key):
            return None

        def find_by_submission(self, submission_id):
            return None

    def _attempt(self):
        attempts = AttemptStore(store=self._Boom())
        return attempts.start(key=KEY, submission_id="sub-1")

    def test_a_failed_publication_does_not_mark_the_attempt_completed(self):
        attempt = self._attempt()
        with pytest.raises(PublicationError):
            attempt.complete(RESULT)
        assert attempt.state == STATE_RUNNING
        assert attempt.result is None

    def test_a_failed_publication_still_marks_the_attempt_failed(self):
        """**The failure path deliberately goes the other way.**

        On the success path an unpublished attempt must not be left
        terminal -- there is a result worth retrying for, and claiming
        durability that does not exist is the whole defect. Here there is
        no result and nothing to retry: the attempt *did* fail. Leaving it
        `running` because the *record* of the failure did not land would
        report an in-flight job to a coordinator that would wait forever.

        The error is still raised, so the caller can decide; the state is
        set regardless.
        """
        attempt = self._attempt()
        with pytest.raises(PublicationError):
            attempt.fail("PRICING_FAILED")
        assert attempt.state == STATE_FAILED
        assert attempt.reason == "PRICING_FAILED"

    def test_a_swallowed_publication_failure_does_not_strand_the_attempt(self):
        """What `eod_routes._record_failure` actually does: swallow and
        return an error naming the real cause. The attempt must not be
        left discoverable as `running` afterwards."""
        attempts = AttemptStore(store=self._Boom())
        attempt = attempts.start(key=KEY, submission_id="sub-1")
        try:
            attempt.fail("TERMS_ARTIFACT_UNUSABLE")
        except PublicationError:
            pass  # exactly what the route does

        assert attempts.lookup(KEY).state == STATE_FAILED

    def test_the_attempt_can_still_be_completed_once_the_store_recovers(self, tmp_path):
        """The point of not marking it terminal: the retry must be allowed.

        Under the old ordering the immutability guard fired on the second
        call (`attempt ... is already completed`), so a transient store
        failure permanently bricked the attempt."""
        attempts = AttemptStore(store=self._Boom())
        attempt = attempts.start(key=KEY, submission_id="sub-1")
        with pytest.raises(PublicationError):
            attempt.complete(RESULT)

        # Store recovers; the same attempt completes for real.
        working = ResultStore(tmp_path / "recovered")
        attempt._store = working
        attempt.complete(RESULT)

        assert attempt.state == STATE_COMPLETED
        assert working.lookup(KEY)["result"]["bundleId"] == "B1"

    def test_an_unpublished_attempt_is_not_reported_as_completed_in_process(self):
        """The divergence this class exists to prevent, stated directly."""
        attempts = AttemptStore(store=self._Boom())
        attempt = attempts.start(key=KEY, submission_id="sub-1")
        with pytest.raises(PublicationError):
            attempt.complete(RESULT)

        assert attempts.lookup(KEY).state == STATE_RUNNING
        assert attempts.get(attempt.attempt_id).state == STATE_RUNNING
