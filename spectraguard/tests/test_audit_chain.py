"""Unit tests for cryptographic audit log hash chain validation."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base
from backend.records import crud, models


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_audit_log_hash_chain(db_session):
    # Append 3 audit log entries
    e1 = crud.append_audit_log(db_session, user="admin", action="start", detail="System boot")
    e2 = crud.append_audit_log(db_session, user="operator", action="calibration", detail="Dark ref")
    e3 = crud.append_audit_log(db_session, user="qa", action="lot_hold", detail="Held lot Lot-001")

    logs = crud.list_audit_log(db_session)
    # Order returned is descending by ts/id
    logs_asc = list(reversed(logs))

    # Verify first entry has prev_hash of 64 zeros
    assert logs_asc[0].prev_hash == "0" * 64

    # Verify each subsequent entry's prev_hash matches previous entry's entry_hash
    for i in range(1, len(logs_asc)):
        assert logs_asc[i].prev_hash == logs_asc[i - 1].entry_hash
