import dotenv
import pendulum
from sqlalchemy import asc, cast
from sqlalchemy import create_engine, select, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, BigInteger
from config import Settings
from models import MessageSql

house_alias = "beech"
message_type = "atn.bid"
start_ms = pendulum.datetime(2025, 11, 25, 10, 30, tz='America/New_York').timestamp()*1000
end_ms = pendulum.datetime(2025, 11, 25, 17, 30, tz='America/New_York').timestamp()*1000

stmt = select(MessageSql).filter(
    MessageSql.message_type_name == message_type,
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(int(end_ms), BigInteger),
    MessageSql.message_persisted_ms >= cast(int(start_ms), BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
result = session.execute(stmt)
messages = result.scalars().all()

print(f"Found {len(messages)} messages")

for m in [messages[0]]:
    print(m.from_alias)
    print(pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York'))
    print(m.payload)

# import matplotlib.pyplot as plt
# times = [pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York') for m in messages]
# plt.scatter(times, [1]*len(messages))
# plt.show()
