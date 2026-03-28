import os
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


vis_sessionmaker = None
vis_sessionmaker_lock = threading.Lock()
def get_sessionmaker():
    global vis_sessionmaker
    if vis_sessionmaker is None:
        with vis_sessionmaker_lock:
            if vis_sessionmaker is None:
                url = os.getenv('VIS2_DB_URL')
                engine = create_engine(url, echo=True)
                vis_sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return vis_sessionmaker

def get_db():
    sessionmaker = get_sessionmaker()
    db = sessionmaker()
    try:
        yield db  # This session is shared with the route and its sub-dependencies
    finally:
        db.close() # Automatic cleanup after the request finishes