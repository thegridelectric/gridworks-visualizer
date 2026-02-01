import dotenv
import pendulum
from sqlalchemy import asc, cast
from sqlalchemy import create_engine, select, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, BigInteger
from config import Settings
from models import MessageSql
import matplotlib.pyplot as plt

house_alias = "beech"
message_type = "weather.forecast"
start_ms = pendulum.datetime(2026, 1, 31, 0, 0, 0, tz='America/New_York').timestamp()*1000
end_ms = pendulum.datetime(2026, 2, 2, 0, 0, 0, tz='America/New_York').timestamp()*1000

stmt = select(MessageSql).filter(
    MessageSql.message_type_name == message_type,
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada",
    MessageSql.message_created_ms <= cast(int(end_ms), BigInteger),
    MessageSql.message_created_ms >= cast(int(start_ms), BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
result = session.execute(stmt)
messages = result.scalars().all()

print(f"Found {len(messages)} messages")

timestamps = []
oat_f = []
for m in messages:
    print(pendulum.from_timestamp(m.message_created_ms/1000, tz='America/New_York'))
    print(m.payload['OatF'][0])
    timestamps.append(pendulum.from_timestamp(m.message_created_ms/1000, tz='America/New_York'))
    oat_f.append(m.payload['OatF'][0])

import pandas as pd

df = pd.DataFrame({
    "timestamps": timestamps,
    "oat": oat_f
})
df["timestamps"] = pd.to_datetime(df["timestamps"])
df.to_csv("timestamps_oat.csv", index=False)



import matplotlib.pyplot as plt
plt.plot(timestamps, oat_f)
plt.show()
    
# print("")
# print(messages[0].payload['Ha1Params'])

# import matplotlib.pyplot as plt
# times = [pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York') for m in messages]
# plt.scatter(times, [1]*len(messages))
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('/Users/thomas/Downloads/beech_30s_2025-12-21-05_00-2025-12-23-05_00.csv', header=1)
# df['timestamps'] = pd.to_datetime(df['timestamps'])
# plt.figure(figsize=(11, 4))
# plt.plot(df['timestamps'], df['hp-lwt'])
# plt.show()