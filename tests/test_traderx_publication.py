"""
Crash-safe publication and the durable result store (`engine.traderx.publication`,
I-08). These test what survives a crash; the state machine itself is covered by
test_traderx_routes.py, which passes against a memory-only store.

Crash windows, each reproduced by leaving the store in the state that crash would:

  1. between artifact write and manifest publish: lookup finds nothing, in particular no
     partial result;
  2. after publish: lookup finds the complete result;
  3. between publish and pointer advance: lookup still finds it, via the scan (the manifest,
     not the pointer, is the commit point);
  4. restart: completed attempts survive; running ones are reported as unknown.

The state is constructed directly (e.g. deleting the pointer after a publication) because a
real kill cannot be landed in the window between rename and pointer write on demand.
"""
import json
from pathlib import Path

import pytest

from engine.traderx.publication import (
    PUBLICATION_SCHEMA,
    PublicationError,
    ResultStore,
    default_store_root,
)
from engine.traderx.workload import (
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
    """Filesystem conditions the protocol's guarantees depend on."""

    def test_temp_directory_is_a_sibling_of_attempts(self, store):
        """`os.replace` is atomic only within one filesystem, so tmp/ sits beside
        attempts/."""
        assert (store.root / "tmp").parent == (store.root / "attempts").parent

    def test_publication_leaves_no_temp_files_behind(self, store):
        _publish(store, "att-1")
        assert list((store.root / "tmp").iterdir()) == []

    def test_pointer_filename_contains_no_colon(self, store):
        """A workload key is `sha256:<hex>`; `:` is not legal in a Windows filename, so the
        key is digested."""
        _publish(store, "att-1")
        names = [p.name for p in (store.root / "pointers").iterdir()]
        assert names and all(":" not in name for name in names)

    def test_manifest_records_its_own_schema(self, store):
        """A reader can refuse an unfamiliar layout rather than guess."""
        manifest = _publish(store, "att-1")
        assert manifest["publicationSchema"] == PUBLICATION_SCHEMA


class TestCrashBeforeManifestPublish:
    """Window 1: nothing partial is discoverable."""

    def test_staged_bytes_are_not_discoverable(self, store):
        """A complete-looking manifest in tmp/ is not published."""
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
        """A torn manifest under attempts/ reads as absent, not as a partial result."""
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
        """An unreadable file is skipped, so one bad file cannot break every lookup."""
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
    """Window 3: a crash leaves a published result with a stale pointer. Trusting the
    pointer would answer `UNKNOWN_WORKLOAD` for finished work; every test here fails
    against a pointer-only lookup."""

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
        """The scan cost is paid once per crash."""
        _publish(store, "att-1")
        self._drop_pointer(store)
        assert list((store.root / "pointers").iterdir()) == []
        store.lookup(KEY)
        assert len(list((store.root / "pointers").iterdir())) == 1

    def test_a_stale_pointer_does_not_hide_a_newer_result(self, store):
        """A second publication commits but dies before advancing the pointer, which still
        names a valid older attempt; the scan must find the newer one."""
        _publish(store, "att-old")
        pointer_after_first = store._pointer_path(KEY).read_bytes()
        _publish(store, "att-new")
        store._pointer_path(KEY).write_bytes(pointer_after_first)
        assert store.lookup(KEY)["attemptId"] == "att-new"

    def test_the_pointer_never_moves_backwards(self, store):
        """A late publication or reconciling scan never points the cache at a superseded
        attempt."""
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
        """A pointer naming another workload's attempt is not trusted (it would serve a
        result from different inputs)."""
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
    """Lookup serves the most recent successful attempt, never a failed or in-flight one."""

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
        """A running attempt has no result to commit and is never published."""
        with pytest.raises(PublicationError, match="only terminal"):
            store.publish(
                attempt_id="att-1", workload_key=KEY, state=STATE_RUNNING
            )

    def test_a_refused_publication_writes_nothing(self, store):
        with pytest.raises(PublicationError):
            store.publish(attempt_id="att-1", workload_key=KEY, state=STATE_RUNNING)
        assert list((store.root / "attempts").iterdir()) == []


class TestStep2VerifiesWhatWasActuallyWritten:
    """Step 2 verifies the bytes actually written, not those meant to be written."""

    def test_a_corrupted_write_is_refused_and_publishes_nothing(self, store, monkeypatch):
        """The filesystem returning different bytes than were written is refused."""
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
        """The error names both digests (truncation vs substitution)."""
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
        """Manifests differing only in dict order produce the same bytes and hash."""
        _publish(store, "att-1", submission_id="s1")
        first = (store.root / "attempts" / "att-1.json").read_bytes()
        store.clear()
        _publish(store, "att-1", submission_id="s1")
        assert (store.root / "attempts" / "att-1.json").read_bytes() == first


class TestRestartSurvival:
    """A new `AttemptStore` over the same root is what a restart produces."""

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
        """Submission idempotency survives a restart, so a retried lost response does not
        start a second computation."""
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
        """A running attempt is never published, so after a restart it reads as unknown; the
        workload key makes the resubmission the same computation."""
        live = AttemptStore(store=store)
        live.start(KEY)

        restarted = AttemptStore(store=ResultStore(store.root))
        assert restarted.lookup(KEY) is None

    def test_a_rehydrated_attempt_cannot_be_completed_again(self, store):
        """Immutability survives the restart."""
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        recovered = restarted.get(attempt.attempt_id)
        with pytest.raises(ValueError, match="immutable"):
            recovered.complete({"resultSchema": "different"})

    def test_a_result_published_after_a_restart_is_the_most_recent_one(self, store):
        """Publication order survives the process that assigned it. A process-local counter
        restarts at zero, so a later run's attempt would claim to predate the earlier one
        and lookup would serve the older result as the most recent. This is the one test
        that separates a disk-recovered sequence from a process-local counter.
        """
        _publish(store, "att-zzz-first")

        # A restart: a new store object over the same root.
        restarted_store = ResultStore(store.root)
        restarted_store.publish(
            attempt_id="att-aaa-second",
            workload_key=KEY,
            state=STATE_COMPLETED,
            result={"resultSchema": "jax.eod-result.v1", "bundleId": "NEW", "items": []},
        )

        # Attempt ids chosen so the tie-break points the wrong way: with a process-local
        # counter both are sequence 0 and the id tie-break prefers `att-zzz-first`. (With
        # uuid4 ids the bug would show about half the time.)
        assert restarted_store.lookup(KEY)["attemptId"] == "att-aaa-second"
        assert restarted_store.lookup(KEY)["result"]["bundleId"] == "NEW"

        # A third reader, with no memory, agrees.
        assert ResultStore(store.root).lookup(KEY)["attemptId"] == "att-aaa-second"

    def test_the_sequence_high_water_mark_is_recovered_when_lost(self, store):
        """Losing the high-water file costs a scan, not the ordering."""
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
    """Memory is authoritative for running state, the store across restarts."""

    def test_a_running_retry_does_not_mask_a_published_success(self, store):
        live = AttemptStore(store=store)
        done = live.start(KEY, submission_id="s1")
        done.complete(RESULT)
        live.start(KEY, submission_id="s2")  # a second, still running
        assert live.lookup(KEY).attempt_id == done.attempt_id

    def test_a_published_success_outranks_an_in_memory_failure(self, store):
        """A success on disk outranks a later failure in memory."""
        first = AttemptStore(store=store)
        ok = first.start(KEY)
        ok.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        bad = restarted.start(KEY)
        bad.fail("boom")
        assert restarted.lookup(KEY).attempt_id == ok.attempt_id

    def test_memory_answers_before_the_store_is_consulted(self, store):
        """A memory hit returns the same object callers hold, not a copy from disk.

        This pins the precedence (every path checks memory first). It does not test
        `_rehydrate`'s guard against replacing a live attempt, which current call paths
        cannot reach; the guard is kept for a future caller that scans first.
        """
        live = AttemptStore(store=store)
        attempt = live.start(KEY, submission_id="sub-live")
        attempt.complete(RESULT)

        assert live.get(attempt.attempt_id) is attempt
        assert live.lookup(KEY) is attempt
        assert live.start(KEY, submission_id="sub-live") is attempt

    def test_a_rehydrated_attempt_does_not_republish(self, store):
        """An attempt read back from disk has no store, so it cannot rewrite its
        immutable manifest."""
        live = AttemptStore(store=store)
        attempt = live.start(KEY)
        attempt.complete(RESULT)

        restarted = AttemptStore(store=ResultStore(store.root))
        recovered = restarted.get(attempt.attempt_id)
        assert recovered._store is None


class TestMemoryOnlyRemainsTheDefault:
    """With no store argument, `AttemptStore()` is memory-only, as before."""

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
        """The default root is not the working directory."""
        monkeypatch.delenv("JAX_EOD_STORE_ROOT", raising=False)
        assert default_store_root() != Path.cwd()


class TestConcurrentPublication:
    """The sequence read-then-increment is locked (the store serves a thread pool)."""

    def test_concurrent_publications_get_distinct_sequences(self, store):
        """Unlocked, 30 concurrent publications produced 2 distinct sequences, turning "most
        recent" into an attempt-id tie-break."""
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
        """`lookup` holds `_lock` and `publish` takes `_sequence_lock`; they never wait on
        each other."""
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
    """A failed rename is a `PublicationError`, so the route answers
    `RESULT_NOT_PUBLISHED` rather than an opaque 500."""

    def test_a_failed_rename_raises_publication_error(self, store, monkeypatch):
        def refuse(src, dst):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr("engine.traderx.publication.os.replace", refuse)
        with pytest.raises(PublicationError, match="could not be published"):
            _publish(store, "att-1")

    def test_a_failed_rename_leaves_nothing_behind(self, store, monkeypatch):
        def refuse(src, dst):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("engine.traderx.publication.os.replace", refuse)
        with pytest.raises(PublicationError):
            _publish(store, "att-1")
        monkeypatch.undo()

        assert store.read_attempt("att-1") is None
        assert store.lookup(KEY) is None
        assert list((store.root / "tmp").iterdir()) == []


class TestAFailedPublicationLeavesNoTerminalAttemptInMemory:
    """The manifest is the commit point for the in-memory attempt too.

    `complete()` once set the state and then published; when publication failed, the
    attempt was completed in memory with nothing on disk, and the immutability guard then
    refused the retry. Publishing first means either both happened or neither did.
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
        """The failure path goes the other way: the attempt did fail, so it is marked failed
        even if that record does not land (leaving it running would make a coordinator wait
        forever). The error is still raised."""
        attempt = self._attempt()
        with pytest.raises(PublicationError):
            attempt.fail("PRICING_FAILED")
        assert attempt.state == STATE_FAILED
        assert attempt.reason == "PRICING_FAILED"

    def test_a_swallowed_publication_failure_does_not_strand_the_attempt(self):
        """As `traderx_routes._record_failure` does (swallowing the error), the attempt is not
        left running."""
        attempts = AttemptStore(store=self._Boom())
        attempt = attempts.start(key=KEY, submission_id="sub-1")
        try:
            attempt.fail("TERMS_ARTIFACT_UNUSABLE")
        except PublicationError:
            pass  # exactly what the route does

        assert attempts.lookup(KEY).state == STATE_FAILED

    def test_the_attempt_can_still_be_completed_once_the_store_recovers(self, tmp_path):
        """Because it was not marked terminal, the retry is allowed once the store
        recovers."""
        attempts = AttemptStore(store=self._Boom())
        attempt = attempts.start(key=KEY, submission_id="sub-1")
        with pytest.raises(PublicationError):
            attempt.complete(RESULT)

        # The store recovers; the same attempt completes.
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
