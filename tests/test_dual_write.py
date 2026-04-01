class _FakeDB:
    def __init__(self):
        self.events = []

    def upsert_job(self, *args, **kwargs):
        self.events.append(("upsert_job", args, kwargs))
        return True, False

    def count_jobs(self):
        self.events.append(("count_jobs", (), {}))
        return 3


def test_dual_write_database_mirrors_writes_and_delegates_reads():
    from src.mcf.dual_write import DualWriteDatabase

    primary = _FakeDB()
    secondary = _FakeDB()
    db = DualWriteDatabase(primary, secondary)

    assert db.upsert_job("job") == (True, False)
    assert db.count_jobs() == 3

    assert primary.events[0][0] == "upsert_job"
    assert secondary.events[0][0] == "upsert_job"
    assert primary.events[1][0] == "count_jobs"
    assert len(secondary.events) == 1


class _FakeSessionDB:
    def __init__(self, next_session_id: int = 1, next_historical_id: int = 1):
        self.next_session_id = next_session_id
        self.next_historical_id = next_historical_id
        self.session_rows = {}
        self.historical_rows = {}

    def create_session(self, search_query, total_jobs, session_id=None):
        resolved_id = self.next_session_id if session_id is None else session_id
        self.session_rows[resolved_id] = {
            "search_query": search_query,
            "total_jobs": total_jobs,
            "completed": False,
        }
        self.next_session_id = max(self.next_session_id, resolved_id + 1)
        return resolved_id

    def update_session(self, session_id, fetched_count, current_offset):
        row = self.session_rows[session_id]
        row["fetched_count"] = fetched_count
        row["current_offset"] = current_offset

    def complete_session(self, session_id):
        self.session_rows[session_id]["completed"] = True

    def create_historical_session(self, year, start_seq, end_seq=None, session_id=None, conn=None):
        del conn
        resolved_id = self.next_historical_id if session_id is None else session_id
        self.historical_rows[resolved_id] = {
            "year": year,
            "start_seq": start_seq,
            "end_seq": end_seq,
            "completed": False,
        }
        self.next_historical_id = max(self.next_historical_id, resolved_id + 1)
        return resolved_id

    def update_historical_progress(
        self, session_id, current_seq, jobs_found, jobs_not_found, consecutive_not_found=0, end_seq=None, conn=None
    ):
        del conn
        row = self.historical_rows[session_id]
        row["current_seq"] = current_seq
        row["jobs_found"] = jobs_found
        row["jobs_not_found"] = jobs_not_found
        row["consecutive_not_found"] = consecutive_not_found
        row["end_seq"] = end_seq if end_seq is not None else row["end_seq"]

    def complete_historical_session(self, session_id, conn=None):
        del conn
        self.historical_rows[session_id]["completed"] = True


def test_dual_write_database_preserves_session_ids_across_backends():
    from src.mcf.dual_write import DualWriteDatabase

    primary = _FakeSessionDB(next_session_id=3, next_historical_id=7)
    secondary = _FakeSessionDB(next_session_id=101, next_historical_id=205)
    db = DualWriteDatabase(primary, secondary)

    session_id = db.create_session("python", 42)
    db.update_session(session_id, 10, 20)
    db.complete_session(session_id)

    historical_id = db.create_historical_session(2026, 1, 99)
    db.update_historical_progress(historical_id, 50, 12, 3, consecutive_not_found=1, end_seq=120)
    db.complete_historical_session(historical_id)

    assert session_id == 3
    assert primary.session_rows[session_id] == secondary.session_rows[session_id]
    assert historical_id == 7
    assert primary.historical_rows[historical_id] == secondary.historical_rows[historical_id]
