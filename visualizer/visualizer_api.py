import io
import gc
import json
import os
import time
import uuid
import pytz
import dotenv
import pendulum
import uvicorn
import zipfile
import traceback
import numpy as np
import pandas as pd
import httpx
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union
import asyncio
import async_timeout
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import asc, or_, and_, desc, cast
from sqlalchemy import create_engine, MetaData, Table, select, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.future import select
from jose import JWTError, jwt
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from config import Settings
from models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo import Flo, DGraphVisualizer

#from v2.routers import sessions as v2_sessions
import v2.routers.synced_readings_bundle as v2_synced_readings_bundle

print("Starting API...")

CSV_SAMPLING = True

# ------------------------------
# Pydantic models
# ------------------------------

class Prices(BaseModel):
    unix_s: List[float]
    lmp: List[float]
    dist: List[float]
    energy: List[float]
    
class BaseRequest(BaseModel):
    house_alias: Union[str, List[str]]
    password: str
    unique_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    def __hash__(self):
        return hash(self.unique_id)

class DataRequest(BaseRequest):
    start_ms: int
    end_ms: int
    selected_channels: List[str]
    confirm_with_user: Optional[bool] = False
    darkmode: Optional[bool] = False

class CsvRequest(DataRequest):
    timestep: int

class MessagesRequest(DataRequest):
    selected_message_types: List[str]

class FloRequest(BaseRequest):
    time_ms: int

class ElectricityUseRequest(BaseModel):
    selected_short_aliases: List[str]
    darkmode: Optional[bool] = False
    start_ms: int
    end_ms: int

class Token(BaseModel):
    username: str
    roles: dict[str, str]
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    username: str
    email: Optional[str] = None
    is_active: Optional[bool] = None

class House(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    short_alias: Optional[str] = None
    address: Optional[dict] = None
    primary_contact: Optional[dict] = None
    secondary_contact: Optional[dict] = None
    hardware_layout: Optional[dict] = None
    unique_id: int
    g_node_alias: Optional[str] = None
    alert_status: Optional[dict] = None
    representation_status: Optional[str] = None
    scada_ip_address: str
    scada_git_commit: str
    house_parameters: Optional[dict] = None

class ScadaUpdateRequest(BaseModel):
    selected_short_aliases: List[str]
    update_packages: bool = False

# ------------------------------
# Backoffice database setup
# ------------------------------

env_file = dotenv.find_dotenv()
print(f'Loading .env file from {env_file}')
dotenv.load_dotenv(env_file)
settings = Settings(_env_file=env_file)

engine_gbo = create_engine(settings.gbo_db_url_no_async.get_secret_value())
gbo_secret_key = settings.secret_key.get_secret_value()
gbo_algorithm = "HS256"
gbo_access_token_expire_minutes = int(7*24*60)
users = Table('users', MetaData(), autoload_with=engine_gbo)
user_roles = Table('user_roles', MetaData(), autoload_with=engine_gbo)
homes = Table('homes', MetaData(), autoload_with=engine_gbo)
hourly_electricity = Table('hourly_electricity', MetaData(), autoload_with=engine_gbo)
gbo_pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
    bcrypt__ident="2b"
)
gbo_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
Session: Session = sessionmaker(bind=engine_gbo)

# ------------------------------
# Database functions
# ------------------------------

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return gbo_pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, gbo_secret_key, algorithm=gbo_algorithm)
    return encoded_jwt

async def get_current_user(token: str = Depends(gbo_oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, gbo_secret_key, algorithms=[gbo_algorithm])
        username: str = payload.get("sub")
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.execute(users.select().where(users.c.username == token_data.username)).first()
    if user is None:
        raise credentials_exception
    return user


class VisualizerApi():
    def __init__(self):
        self.settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_async_engine(self.settings.db_url.get_secret_value())
        self.running_locally = self.settings.running_locally
        self.AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        self.admin_user_password = self.settings.visualizer_api_password.get_secret_value()
        self.timezone_str = 'America/New_York'
        self.timeout_seconds = None if self.running_locally else 5*60
        self.top_states_order = ['LocalControl', 'LeafTransactiveNode', 'Dormant']
        self.lc_states_order = [
            'HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 
            'HpOnStoreCharge', 'StratBoss', 'Initializing', 'Dormant', 'EverythingOff'
            ]
        self.la_states_order = self.lc_states_order.copy()
        self.whitewire_threshold_watts = {'beech': 100, 'elm': 1, 'default': 20}
        self.zone_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*3
        self.time_spent_reducing_data = 0
        self.threshold_per_channel = {
            'hp-lwt': 0.2*1000, #degCx1000
            'hp-ewt': 0.2*1000, #degCx1000
            'hp-odu-pwr': 0.1*1000, #kWx1000
            'hp-idu-pwr': 0.1*1000, #kWx1000
            'oil-boiler-pwr': 0.2*100, #kWx100
            'primary-flow': 0.25*100, #GPMx100
            'primary-pump-pwr': 0.3*10, #kWx100
            'dist-swt': 1*1000, #degCx1000
            'dist-rwt': 1*1000, #degCx1000
            'dist-flow': 0.2*100, #GPMx100
            'dist-pump-pwr': 0.3*10, #Wx10
            'oat': 0.5*1000, #degCx1000
            'buffer-depths': 0.2*100, #degFx100
            'tank-depths': 0.2*100, #degFx100
            'buffer-hot-pipe': 0.2*1000, #degCx1000
            'buffer-cold-pipe': 0.2*1000, #degCx1000
            'store-hot-pipe': 0.2*1000, #degCx1000
            'store-cold-pipe': 0.2*1000, #degCx1000
            'store-flow': 0.1*100, #GPMx100
            'store-pump-pwr': 1*10, #kWx100
            'zone-temp': 0.5*1000, #degFx1000
            'zone-set': 0.5*1000, #degFx1000
            'zone': 0,
        }
        self.data: dict[BaseRequest, dict] = {}
        self.timestamp_min_max: dict[BaseRequest, dict[str, datetime]] = {}
        print(f"Running API {'locally' if self.running_locally else 'on EC2'}")

    def start(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"]
        )
        # Compress large JSON payloads (e.g. /plots) when the client supports gzip.
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        self.app.post("/login", response_model=Token)(self.login)
        self.app.post("/logout")(self.logout)
        self.app.get("/google-maps-api-key")(self.get_google_maps_api_key)
        self.app.get("/me", response_model=User)(self.read_current_user)
        self.app.get("/homes", response_model=list[House])(self.get_homes)
        self.app.post("/electricity-use")(self.get_electricity_use)
        self.app.post("/electricity-use-csv")(self.get_electricity_use_csv)
        self.app.post("/plots")(self.get_plots)
        self.app.post("/csv")(self.get_csv)
        self.app.post("/messages")(self.get_messages)
        self.app.post("/flo")(self.get_flo)
        self.app.post("/update-scada-code")(self.update_scada_code)

        self.app.include_router(v2_synced_readings_bundle.router)

        uvicorn.run(self.app, host="0.0.0.0", port=8000)

    def to_datetime(self, time_ms, pendulum_format=False):
        if pendulum_format:
            return pendulum.from_timestamp(time_ms / 1000, tz=self.timezone_str)
        else:
            return datetime.fromtimestamp(time_ms / 1000, tz=pytz.timezone(self.timezone_str))

    def to_fahrenheit(self, t):
        return t*9/5+32
    
    def to_hex(self, rgba):
        r, g, b, a = (int(c * 255) for c in rgba)
        return f'#{r:02x}{g:02x}{b:02x}'

    async def login(self, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
        user = db.execute(users.select().where(users.c.username == form_data.username)).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=gbo_access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        # Update last login with timezone-aware datetime
        db.execute(
            users.update().where(users.c.username == user.username).values(
                last_login=datetime.now(timezone.utc)
            )
        )
        db.commit()

        role_rows = db.execute(
            user_roles.select().where(user_roles.c.username == user.username)
        ).fetchall()
        roles: dict[str, str] = {}
        for row in role_rows:
            role_val = row.role
            if hasattr(role_val, "value"):
                role_val = role_val.value
            roles[str(role_val)] = row.installation

        data = {
            "username": user.username,
            "roles": roles,
            "access_token": access_token,
            "token_type": "bearer",
        }
        return data

    async def logout(self, current_user = Depends(get_current_user)):
        return {"message": "Successfully logged out"}

    async def read_current_user(self, current_user = Depends(get_current_user)):
        return User(username=current_user.username)

    async def get_google_maps_api_key(self, current_user = Depends(get_current_user)):
        return {"api_key": settings.google_maps_api_key.get_secret_value()}

    def add_internet_down_highlights(self, fig: go.Figure, request: BaseRequest):
        for period_start, period_end in self.data[request].get('late_persistence_periods', []):
            fig.add_vrect(
                x0=period_start, x1=period_end,
                fillcolor="red", opacity=0.15,
                layer="below", line_width=0,
            )
    
    def reduce_data_size(self, channel_data, channel_name, max_timestamp):
        if 'buffer-depth' in channel_name and 'micro' not in channel_name and 'device' not in channel_name:
            channel_name = 'buffer-depths'
        if 'tank' in channel_name and 'depth' in channel_name and 'micro' not in channel_name and 'device' not in channel_name:
            channel_name = 'tank-depths'
        if 'zone' in channel_name:
            if '-temp' in channel_name:
                channel_name = 'zone-temp'
            elif '-set' in channel_name:
                channel_name = 'zone-set'
            else:
                channel_name = 'zone'

        if channel_name not in self.threshold_per_channel or not channel_data['values'] or len(channel_data['values']) < 2:
            return channel_data
                
        reduced_times = [channel_data['times'][0]]
        reduced_values = [channel_data['values'][0]]
        for i in range(1, len(channel_data['values'])):
            if abs(channel_data['values'][i] - reduced_values[-1]) >= self.threshold_per_channel[channel_name]:
                reduced_times.append(channel_data['times'][i])
                reduced_values.append(channel_data['values'][i])
        reduced_times.append(channel_data['times'][-1])
        reduced_values.append(channel_data['values'][-1])
        
        # reduced_times.append(max_timestamp)
        # reduced_values.append(reduced_values[-1])
        return {'times': reduced_times, 'values': reduced_values}
    
    def check_request(self, request: BaseRequest):
        if not self.running_locally: 
            MAX_PLOT_DAYS = 3
            MAX_CSV_DAYS = 2
            MAX_MESSAGES_DAYS = 31
            # Max plot time range
            if isinstance(request, DataRequest) and not isinstance(request, CsvRequest): 
                if (request.end_ms-request.start_ms)/1000/3600/24 > MAX_PLOT_DAYS:
                    warning_message = f"Plotting data from more than {MAX_PLOT_DAYS} days is not permitted to prevent the visualizer EC2 instance from crashing."
                    warning_message += "\n\nPlease reduce the query time range, or consider running the visualizer API locally for larger queries."
                    return {"success": False, "message": warning_message, "reload": False}
            
            # Max CSV download time range
            if isinstance(request, CsvRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > MAX_CSV_DAYS:
                warning_message = f"Downloading data from more than {MAX_CSV_DAYS} days is not permitted to prevent the visualizer EC2 instance from crashing."
                warning_message += "\n\nPlease reduce the query time range, or consider running the visualizer API locally for larger queries."
                return {"success": False, "message": warning_message, "reload": False}
            
            # Max messages time range
            if isinstance(request, MessagesRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > MAX_MESSAGES_DAYS:
                warning_message = f"Downloading messages from more than {MAX_MESSAGES_DAYS} days is not permitted to prevent the visualizer EC2 instance from crashing."
                warning_message += "\n\nPlease reduce the query time range, or consider running the visualizer API locally for larger queries."
                return {"success": False, "message": warning_message, "reload": False}
        # else:
        #     if isinstance(request, Union[DataRequest, CsvRequest]) and not request.confirm_with_user:
        #         if (request.end_ms - request.start_ms)/1000/3600/24 > 30:
        #             warning_message = f"That's a lot of data! Are you sure you want to proceed?"
        #             return {"success": False, "message": warning_message, "reload": False, "confirm_with_user": True}
        return None

    async def get_data(self, request: BaseRequest):
        try:
            error = self.check_request(request)
            if error or request.selected_channels==['bids']:
                if error: print(error)
                return error
            
            self.data[request] = {}
            async with self.AsyncSessionLocal() as session:
                import time
                query_start = time.time()
                print("Querying journaldb...")
                
                # Use select() instead of session.query()
                stmt = select(MessageSql).filter(
                    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{request.house_alias}.scada",
                    MessageSql.message_persisted_ms <= cast(int(request.end_ms), BigInteger),
                    or_(
                        and_(
                            or_(
                                MessageSql.message_type_name == "batched.readings",
                                MessageSql.message_type_name == "report",
                            ),
                            MessageSql.message_persisted_ms >= cast(int(request.start_ms), BigInteger),
                        ),
                        and_(
                            MessageSql.message_type_name == "snapshot.spaceheat",
                            MessageSql.message_persisted_ms >= cast(int(request.end_ms - 10*60*1000), BigInteger),
                        ),
                        and_(
                            MessageSql.message_type_name == "weather.forecast",
                            MessageSql.message_persisted_ms >= cast(int(request.start_ms - 24 * 3600 * 1000), BigInteger),
                        )
                    )
                ).order_by(asc(MessageSql.message_persisted_ms))
                
                # Execute the statement asynchronously
                result = await session.execute(stmt)
                all_raw_messages: List[MessageSql] = result.scalars().all()  # Use scalars() to retrieve the data
                
                print(f"\033[91m- Time to query data from journaldb: {round(time.time() - query_start, 1)}s\033[0m")

            if not all_raw_messages:
                warning_message = f"No data found for house '{request.house_alias}' in the selected timeframe."
                return {"success": False, "message": warning_message, "reload": False}
            
            # Process reports
            print(f"Processing data...")
            reports: List[MessageSql] = sorted(
                [x for x in all_raw_messages if x.message_type_name in ['report', 'batched.readings']],
                key = lambda x: x.message_persisted_ms
                )
            self.data[request]['channels'] = {}
            for message in reports:
                # print(f"\nFound report at {self.to_datetime(message.message_persisted_ms)}:")
                # print([x['ChannelName'] for x in message.payload['ChannelReadingList']])
                for channel in message.payload['ChannelReadingList']:
                    if message.message_type_name == 'report':
                        channel_name = channel['ChannelName']
                    elif message.message_type_name == 'batched.readings':
                        for dc in message.payload['DataChannelList']:
                            if dc['Id'] == channel['ChannelId']:
                                channel_name = dc['Name']
                    if not channel['ValueList'] or not channel['ScadaReadTimeUnixMsList']:
                        continue
                    if len(channel['ValueList'])!=len(channel['ScadaReadTimeUnixMsList']):
                        continue
                    if channel_name not in self.data[request]['channels']:
                        self.data[request]['channels'][channel_name] = {'values': [], 'times': []}
                    self.data[request]['channels'][channel_name]['values'].extend(channel['ValueList'])
                    self.data[request]['channels'][channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
            if not self.data[request]['channels']:
                print(f"No data found.")
                return {"success": False, "message": "No data found.", "reload": False}
                
            # Process snapshots
            max_timestamp = max(max(self.data[request]['channels'][channel_name]['times']) for channel_name in self.data[request]['channels'])
            snapshots = sorted(
                    [x for x in all_raw_messages if x.message_type_name=='snapshot.spaceheat'
                    and x.message_persisted_ms >= max_timestamp], 
                    key = lambda x: x.message_persisted_ms
                    )
            for snapshot in snapshots:
                for snap in snapshot.payload['LatestReadingList']:
                    if snap['ChannelName'] in self.data[request]['channels']:
                        self.data[request]['channels'][snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                        self.data[request]['channels'][snap['ChannelName']]['values'].append(snap['Value'])
            # Get minimum and maximum timestamp for plots
            max_timestamp = max(max(self.data[request]['channels'][x]['times']) for x in self.data[request]['channels'])
            min_timestamp = min(min(self.data[request]['channels'][x]['times']) for x in self.data[request]['channels'])

            min_timestamp = max(request.start_ms, min_timestamp)
            max_timestamp = min(request.end_ms, max_timestamp)
            true_max_timestamp = max_timestamp
            # print(f"After edit: {self.to_datetime(min_timestamp)}")
            min_timestamp += -(max_timestamp-min_timestamp)*0.05
            max_timestamp += (max_timestamp-min_timestamp)*0.05
            self.data[request]['min_timestamp'] = self.to_datetime(min_timestamp)
            self.data[request]['max_timestamp'] = self.to_datetime(max_timestamp)

            # Sort values according to time. Keep epoch ms for plot requests
            # to avoid expensive per-point Python datetime conversion.
            total_conversion_time = 0
            for channel_name in self.data[request]['channels'].keys():
                sorted_times_values = sorted(zip(self.data[request]['channels'][channel_name]['times'], self.data[request]['channels'][channel_name]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                self.data[request]['channels'][channel_name]['values'] = list(sorted_values)
                self.data[request]['channels'][channel_name]['times'] = list(sorted_times)

                # Apply data reduction before converting to datetime
                if not isinstance(request, CsvRequest):
                    self.data[request]['channels'][channel_name] = self.reduce_data_size(
                        self.data[request]['channels'][channel_name], 
                        channel_name,
                        true_max_timestamp
                    )  

                # Convert to datetime only for CSV sampling logic.
                if isinstance(request, CsvRequest) and CSV_SAMPLING:
                    conversion_start = time.time()
                    self.data[request]['channels'][channel_name]['times'] = (
                        pd.to_datetime(
                            self.data[request]['channels'][channel_name]['times'],
                            unit='ms',
                            utc=True
                        )
                        .tz_convert(self.timezone_str)
                        .tz_localize(None)
                        .tolist()
                    )
                    total_conversion_time += time.time() - conversion_start
                    print(f"- Time to convert timestamps to datetime: {round(total_conversion_time, 1)}s")    

            # Find all zone channels
            self.data[request]['channels_by_zone'] = {}
            for channel_name in self.data[request]['channels'].keys():
                if 'zone' in channel_name and 'gw-temp' not in channel_name:
                    zone_number = channel_name.split('-')[0]
                    if zone_number not in self.data[request]['channels_by_zone']:
                        self.data[request]['channels_by_zone'][zone_number] = {}
                    if 'state' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['state'] = channel_name
                    elif 'whitewire' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['whitewire'] = channel_name
                    elif 'temp' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['temp'] = channel_name
                    elif 'set' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['set'] = channel_name

            # Relays
            relays = {}
            for message in reports:
                if 'StateList' not in message.payload:
                    continue
                for state in message.payload['StateList']:
                    state_machine_handle = str(state['MachineHandle']).replace('auto.h', 'auto.lc').replace('a.aa', 'ltn.la')
                    if state_machine_handle not in relays:
                        relays[state_machine_handle] = {'times': [], 'values': []}
                    relays[state_machine_handle]['times'].extend(state['UnixMsList'])
                    relays[state_machine_handle]['values'].extend(state['StateList'])

            # Top state
            self.data[request]['top_states'] = {'all': {'times':[], 'values':[]}}
            if 'auto' in relays:
                for t, state in zip(relays['auto']['times'], relays['auto']['values']):
                    if state == 'HomeAlone':
                        state = 'LocalControl'
                    if state == 'Atn':
                        state = 'LeafTransactiveNode'
                    if state not in self.top_states_order:
                        print(f"Warning: {state} is not a known top state")
                        continue
                    if state not in self.data[request]['top_states']:
                        self.data[request]['top_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['top_states']['all']['times'].append(t)
                    self.data[request]['top_states']['all']['values'].append(self.top_states_order.index(state))
                    self.data[request]['top_states'][state]['times'].append(t)
                    self.data[request]['top_states'][state]['values'].append(self.top_states_order.index(state))
            if "Dormant" in self.data[request]['top_states']:
                self.data[request]['top_states']['Admin'] = self.data[request]['top_states']['Dormant']
                del self.data[request]['top_states']['Dormant']

            # Identify periods where messages were persisted 10+ minutes after creation
            late_threshold_ms = 5 * 60 * 1000
            reports_by_created = sorted(
                [m for m in reports if m.message_created_ms is not None],
                key=lambda m: m.message_created_ms
            )
            late_periods = []
            period_start = None
            period_end = None
            period_end_persisted_ms = None  # for single-message periods: use creation->persistence so band is visible
            for m in reports_by_created:
                if (m.message_persisted_ms - m.message_created_ms) > late_threshold_ms:
                    if period_start is None:
                        period_start = m.message_created_ms
                    period_end = m.message_created_ms
                    period_end_persisted_ms = m.message_persisted_ms
                else:
                    if period_start is not None:
                        # If zero-width (single message), show band from creation to persistence
                        end_ms = period_end_persisted_ms if period_start == period_end else period_end
                        late_periods.append((self.to_datetime(period_start), self.to_datetime(end_ms)))
                        period_start = None
                        period_end = None
                        period_end_persisted_ms = None
            if period_start is not None:
                end_ms = period_end_persisted_ms if period_start == period_end else period_end
                late_periods.append((self.to_datetime(period_start), self.to_datetime(end_ms)))
            self.data[request]['late_persistence_periods'] = late_periods

            # LocalControl state
            self.data[request]['lc_states'] = {'all': {'times':[], 'values':[]}}
            ha_handles = [h for h in relays.keys() if h in ['auto.lc', 'auto.lc.n']]
            for h in [h for h in relays.keys() if 'auto.lc.' in h and 'auto.lc.n' not in h and 'relay' in h]:
                additional_state = h.split('.relay')[0].split('.')[-1]
                if additional_state not in ha_handles:
                    print(f"Adding {additional_state} state")
                    ha_handles.append(additional_state)
            for ha_handle in ha_handles:
                if ha_handle not in ['auto.lc', 'auto.lc.n']:
                    # Find which relay has the minimum first timestamp for this ha_handle
                    relevant_relays = [x for x in relays if ha_handle in x]
                    min_time = None
                    min_relay = None
                    for x in relevant_relays:
                        if relays[x]['times']:
                            first_time = relays[x]['times'][0]
                            if (min_time is None) or (first_time < min_time):
                                min_time = first_time
                                min_relay = x
                    if min_relay is not None:
                        for t in relays[min_relay]['times']:
                            state = ha_handle
                            if state not in self.data[request]['lc_states']:
                                self.data[request]['lc_states'][state] = {'times':[], 'values':[]}
                            self.data[request]['lc_states']['all']['times'].append(t)
                            self.data[request]['lc_states']['all']['values'].append(self.lc_states_order.index('Initializing'))
                            self.data[request]['lc_states'][state]['times'].append(t)
                            self.data[request]['lc_states'][state]['values'].append(self.lc_states_order.index('Initializing'))
                    continue
                for t, state in zip(relays[ha_handle]['times'], relays[ha_handle]['values']):
                    if state == 'HpOn':
                        state = 'HpOnStoreOff'
                    if state == 'HpOff':
                        state = 'HpOffStoreOff'
                    if state not in self.lc_states_order:
                        print(f"Warning: {state} is not a known HA state")
                        continue
                    if state not in self.data[request]['lc_states']:
                        self.data[request]['lc_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['lc_states']['all']['times'].append(t)
                    self.data[request]['lc_states']['all']['values'].append(self.lc_states_order.index(state))
                    self.data[request]['lc_states'][state]['times'].append(t)
                    self.data[request]['lc_states'][state]['values'].append(self.lc_states_order.index(state))

            # LeafAlly state
            self.data[request]['la_states'] = {'all': {'times':[], 'values':[]}}
            if 'ltn.la' in relays:
                for t, state in zip(relays['ltn.la']['times'], relays['ltn.la']['values']):
                    if state == 'HpOn':
                        state = 'HpOnStoreOff'
                    if state == 'HpOff':
                        state = 'HpOffStoreOff'
                    if state not in self.la_states_order:
                        print(f"Warning: {state} is not a known LA state")
                        continue
                    if state not in self.data[request]['la_states']:
                        self.data[request]['la_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['la_states']['all']['times'].append(t)
                    self.data[request]['la_states']['all']['values'].append(self.la_states_order.index(state))
                    self.data[request]['la_states'][state]['times'].append(t)
                    self.data[request]['la_states'][state]['values'].append(self.la_states_order.index(state))

            # Weather forecasts
            weather_forecasts: List[MessageSql] = []
            if isinstance(request, DataRequest):
                weather_forecasts = sorted(
                    [x for x in all_raw_messages if x.message_type_name=='weather.forecast'], 
                    key = lambda x: x.message_persisted_ms
                    )
            self.data[request]['weather_forecasts'] = weather_forecasts.copy()
            # print(f"Time to process data: {round(time.time() - process_start, 1)} seconds")
            return None
        except Exception as e:
            print(f"An error occurred in get_data():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred when getting data", "reload": False}
    
    async def get_messages(self, request: MessagesRequest):
        print("Recieved message request")
        try:
            error = self.check_request(request)
            if error:
                print(error)
                return error
            async with async_timeout.timeout(self.timeout_seconds):
                print("Querying journaldb for messages...")

                async with self.AsyncSessionLocal() as session:
                    if request.house_alias:
                        # Handle both single string and list of strings
                        house_aliases = request.house_alias if isinstance(request.house_alias, list) else [request.house_alias]
                        house_aliases = [x.replace(' ','') for x in house_aliases[0].split(',')]
                        
                        # Build OR conditions for each house alias
                        house_alias_conditions = []
                        for house_alias in house_aliases:
                            house_alias_conditions.extend([
                                MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada",
                                MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada.s2"
                            ])
                        
                        stmt = select(MessageSql).filter(
                            or_(*house_alias_conditions),
                            MessageSql.message_type_name.in_(request.selected_message_types),
                            MessageSql.message_persisted_ms >= request.start_ms,
                            MessageSql.message_persisted_ms <= request.end_ms,
                        ).order_by(asc(MessageSql.message_persisted_ms))
                    else:
                        stmt = select(MessageSql).filter(
                            MessageSql.message_type_name.in_(request.selected_message_types),
                            MessageSql.message_persisted_ms >= request.start_ms,
                            MessageSql.message_persisted_ms <= request.end_ms,
                        ).order_by(asc(MessageSql.message_persisted_ms))

                    result = await session.execute(stmt)
                    messages: List[MessageSql] = result.scalars().all()

                if not messages:
                    print("No messages found.")
                    return {"success": False, "message": f"No data found.", "reload":False}
                
                # Collecting all messages
                levels = {'critical': 1, 'error': 2, 'warning': 3, 'info': 4, 'debug': 5, 'trace': 6}
                sources, pb_types, summaries, details, times_created = [], [], [], [], []
                
                # Problem Events
                sorted_problem_types = sorted(
                    [m for m in messages if m.message_type_name == 'gridworks.event.problem'],
                    key=lambda x: (levels[x.payload['ProblemType']], x.payload['TimeCreatedMs'])
                )
                for message in sorted_problem_types:
                    source = message.payload['Src']
                    if ".scada" in source and source.split('.')[-1] in ['scada', 's2']:
                        source = source.split('.scada')[0].split('.')[-1]
                    sources.append(source)
                    pb_types.append(message.payload['ProblemType'])
                    summaries.append(message.payload['Summary'])
                    details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
                    times_created.append(str(self.to_datetime(message.payload['TimeCreatedMs']).replace(microsecond=0)))
                
                # Glitches
                sorted_glitches = sorted(
                    [m for m in messages if m.message_type_name == 'glitch'],
                    key=lambda x: (levels[str(x.payload['Type']).lower()], x.payload['CreatedMs'])
                )
                for message in sorted_glitches:
                    source = message.payload['FromGNodeAlias']
                    if ".scada" in source and source.split('.')[-1] in ['scada', 's2']:
                        source = source.split('.scada')[0].split('.')[-1]
                    sources.append(source)
                    pb_types.append(str(message.payload['Type']).lower())
                    summaries.append(message.payload['Summary'])
                    details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
                    times_created.append(str(self.to_datetime(message.payload['CreatedMs']).replace(microsecond=0)))
                
                summary_table = {
                    'critical': str(len([x for x in pb_types if x=='critical'])),
                    'error': str(len([x for x in pb_types if x=='error'])),
                    'warning': str(len([x for x in pb_types if x=='warning'])),
                    'info': str(len([x for x in pb_types if x=='info'])),
                    'debug': str(len([x for x in pb_types if x=='debug'])),
                    'trace': str(len([x for x in pb_types if x=='trace'])),
                }
                for key in summary_table.keys():
                    if summary_table[key]=='0':
                        summary_table[key]=''

                return {
                    "Log level": pb_types,
                    "From node": sources,
                    "Summary": summaries,
                    "Details": details,
                    "Time created": times_created,
                    "SummaryTable": summary_table
                }
            
        except asyncio.TimeoutError:
            print("Timed out in get_messages()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_messages():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting messages", "reload": False}
        
    async def get_csv(self, request: CsvRequest):
        try:
            csv_start = time.time()
            print(f"\n=== CSV GENERATION STARTED ===")
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_data(request)
                if error:
                    print(error)
                    return error
                
                # Find the channels to export
                if 'all-data' in request.selected_channels:
                    channels_to_export = list(self.data[request]['channels'].keys())
                else:
                    channels_to_export = []
                    for channel in request.selected_channels:
                        if channel in self.data[request]['channels']:
                            channels_to_export.append(channel)
                        elif channel == 'zone-heat-calls':
                            for c in self.data[request]['channels'].keys():
                                if 'zone' in c:
                                    channels_to_export.append(c)
                        elif channel == 'buffer-depths':
                            for c in self.data[request]['channels'].keys():
                                if 'depth' in c and 'buffer' in c and 'micro' not in c and 'device' not in c:
                                    channels_to_export.append(c)
                        elif channel == 'storage-depths':
                            for c in self.data[request]['channels'].keys():
                                if 'depth' in c and 'tank' in c and 'micro' not in c and 'device' not in c:
                                    channels_to_export.append(c)
                        elif channel == 'relays':
                            for c in self.data[request]['channels'].keys():
                                if 'relay' in c:
                                    channels_to_export.append(c)
                        elif channel == 'zone-heat-calls':
                            for c in self.data[request]['channels'].keys():
                                if 'zone' in c:
                                    channels_to_export.append(c)
                        elif channel == 'store-energy':
                            for c in self.data[request]['channels'].keys():
                                if 'required-energy' in c or 'available-energy':
                                    channels_to_export.append(c)

                # Check the amount of data that will be generated
                num_points = int((request.end_ms - request.start_ms) / (request.timestep * 1000) + 1)
                if not self.running_locally and num_points * len(channels_to_export) > 3600 * 24 * 3 * len(self.data[request]['channels']):
                    error_message = f"This request would generate too many data points ({num_points*len(channels_to_export)})."
                    error_message += "\n\nSuggestions:\n- Increase the time step\n- Reduce the number of channels"
                    error_message += "\n- Reduce the difference between the start and end time"
                    return {"success": False, "message": error_message, "reload": False}


                if CSV_SAMPLING:
                    # Create the timestamps on which the data will be sampled
                    csv_times = np.linspace(request.start_ms, request.end_ms, num_points)
                    csv_times = pd.to_datetime(csv_times, unit='ms', utc=True)
                    csv_times = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times]
                    
                    # Re-sample the data to the desired time step (optimized)
                    print(f"Sampling data with {request.timestep}-second time step...")
                    request_start = time.time()
                    
                    # Ensure target_df is sorted (should be from linspace, but verify)
                    target_df = pd.DataFrame({'times': csv_times}).sort_values('times')
                    csv_data = {'timestamps': list(target_df['times'])}
                    
                    # Use asyncio.gather to run multiple merge_asof operations in parallel
                    async def resample_channel(channel):
                        try:
                            channel_data = self.data[request]['channels'][channel]
                            channel_df = pd.DataFrame({
                                'times': channel_data['times'],
                                'values': channel_data['values']
                            })
                            
                            # Ensure channel_df is sorted by times (required for merge_asof)
                            if not channel_df['times'].is_monotonic_increasing:
                                channel_df = channel_df.sort_values('times')
                            
                            # Remove any duplicate times (keep last value)
                            # This can happen during daylight savings time changes when clocks "fall back"
                            # and the same hour occurs twice (e.g., 2:00 AM EDT -> 1:00 AM EST)
                            if channel_df['times'].duplicated().any():
                                num_duplicates = channel_df['times'].duplicated().sum()
                                channel_df = channel_df.drop_duplicates(subset='times', keep='last').sort_values('times')
                            
                            # Verify both DataFrames are sorted
                            if not target_df['times'].is_monotonic_increasing:
                                raise ValueError(f"target_df times are not sorted!")
                            if not channel_df['times'].is_monotonic_increasing:
                                raise ValueError(f"channel_df for '{channel}' times are not sorted after processing!")
                            
                            sampled = await asyncio.to_thread(
                                pd.merge_asof,
                                target_df,
                                channel_df,
                                on='times',
                                direction='backward'
                            )
                            return channel, list(sampled['values'])
                        except Exception as e:
                            raise Exception(f"Error resampling channel '{channel}': {e}")
                    
                    # Run all channel resampling in parallel
                    results = await asyncio.gather(*[resample_channel(channel) for channel in channels_to_export])
                    
                    for channel, values in results:
                        csv_data[channel] = values

                    print(f"Sampling done in {round(time.time() - request_start, 1)}s.")

                else:
                    csv_data = {}
                    for channel in channels_to_export:
                        csv_data[f'{channel}-timestamps'] = self.data[request]['channels'][channel]['times']
                        csv_data[f'{channel}-values'] = self.data[request]['channels'][channel]['values']
                    
                    max_len = max(len(v) for v in csv_data.values())
                    for col in csv_data:
                        if len(csv_data[col]) < max_len:
                            csv_data[col] = csv_data[col] + [np.nan]*(max_len-len(csv_data[col]))


                df = pd.DataFrame(csv_data)

                # Build file name
                start_date = self.to_datetime(request.start_ms, pendulum_format=True) 
                end_date = self.to_datetime(request.end_ms, pendulum_format=True)
                formatted_start_date = start_date.to_iso8601_string()[:16].replace('T', '-')
                formatted_end_date = end_date.to_iso8601_string()[:16].replace('T', '-')
                filename = f'{request.house_alias}_{request.timestep}s_{formatted_start_date}-{formatted_end_date}.csv'.replace(':','_')

                # Send back as a CSV
                csv_buffer = io.StringIO()
                csv_buffer.write(filename+'\n')
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                csv_content = csv_buffer.getvalue()
                print(f"CSV file size: {round(len(csv_content)/1024/1024, 1)} MB")
                print(f"=== TOTAL TIME: {round(time.time() - csv_start, 1)}s ===\n")
                return StreamingResponse(
                    iter([csv_buffer.getvalue()]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        except asyncio.TimeoutError:
            print("Timed out in get_csv()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_csv():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting CSV", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_flo(self, request: FloRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print("Finding latest FLO run...")
                flo_params_msg = None
                async with self.AsyncSessionLocal() as session:
                    stmt = select(MessageSql).filter(
                        MessageSql.message_type_name == "flo.params.house0",
                        MessageSql.from_alias == f"hw1.isone.me.versant.keene.{request.house_alias}",
                        MessageSql.message_persisted_ms >= request.time_ms - 48*3600*1000,
                        MessageSql.message_persisted_ms <= request.time_ms,
                    ).order_by(desc(MessageSql.message_persisted_ms))
                    result = await session.execute(stmt)
                    flo_params_msg: MessageSql = result.scalars().first()
                
                if not flo_params_msg:
                    print(f"Could not find a FLO run in the 48 hours prior to {self.to_datetime(request.time_ms)}")
                    if os.path.exists('result.xlsx'):
                        os.remove('result.xlsx')
                    return
                print(f"Found FLO run at {self.to_datetime(flo_params_msg.message_persisted_ms)}")

                print("Running FLO and saving analysis to excel...")
                flo_params = FloParamsHouse0(**flo_params_msg.payload)
                g = Flo(flo_params.to_bytes())
                g.solve_dijkstra()
                v = DGraphVisualizer(g)
                v.export_to_excel()
                del g 
                del v
                gc.collect()
                print("Done.")
                
                if os.path.exists('result.xlsx'):
                    return FileResponse(
                        'result.xlsx',
                        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        headers={"Content-Disposition": "attachment; filename=file.xlsx"}
                        )
                else:
                    return {"error": "File not found"}
        except asyncio.TimeoutError:
            print("Timed out in get_flo()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_flo():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting FLO", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")

    async def get_plots(self, request: DataRequest, current_user = Depends(get_current_user)):
        try:
            total_start = time.time()
            print(f"\n=== PLOT GENERATION STARTED ===")
            print(f"{request.house_alias} - {self.to_datetime(request.start_ms)} to {self.to_datetime(request.end_ms)}")
            
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_data(request)
                if error:
                    print(error)
                    return error
                
                plot_start = time.time()
                print("Generating plots...")
                plots = {}
                plots['plot1'] = await self.plot_heatpump(request)
                plots['plot2'] = await self.plot_prices(request)
                plots['plot3'] = await self.plot_distribution(request)
                plots['plot4'] = await self.plot_heatcalls(request)
                plots['plot5'] = await self.plot_zones(request)
                plots['plot6'] = await self.plot_buffer(request)
                plots['plot7'] = await self.plot_storage(request)
                plots['plot8'] = await self.plot_top_state(request)
                plots['plot9'] = await self.plot_ha_state(request)
                plots['plot10'] = await self.plot_aa_state(request)
                plots['plot11'] = await self.plot_weather(request)

                plot_time = time.time() - plot_start
                print(f"- Time to generate all plots: {round(plot_time, 1)}s")

                response = JSONResponse(content={"success": True, "plots": plots})
                content_length = response.headers.get("content-length")
                if content_length:
                    response_size_mb = int(content_length) / 1024 / 1024
                    print(f"Sent JSON plots ({round(response_size_mb, 1)} MB)")
                else:
                    print("Sent JSON plots")

                total_time = time.time() - total_start
                print(f"=== TOTAL TIME: {round(total_time, 1)}s ===\n")

                return response
                
        except asyncio.TimeoutError:
            print("Timed out in get_plots()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_plots():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting plots", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")

    def _fig_to_plot_spec(self, fig, config=None):
        """Convert Plotly figure to JSON-serializable dict for client-side rendering."""
        if config is None:
            config = {'displayModeBar': False, 'staticPlot': False, 'responsive': True}
        fig_dict = json.loads(fig.to_json())
        fig_dict['config'] = config
        return fig_dict

    def _plot_axis_meta(self, request: BaseRequest) -> dict:
        """Shared x-range and late-persistence bands for time-series plots (epoch ms)."""
        min_dt = self.data[request]['min_timestamp']
        max_dt = self.data[request]['max_timestamp']
        x_range_ms = [int(min_dt.timestamp() * 1000), int(max_dt.timestamp() * 1000)]
        late_ms = [
            [int(p0.timestamp() * 1000), int(p1.timestamp() * 1000)]
            for p0, p1 in self.data[request].get('late_persistence_periods', [])
        ]
        return {'x_range_ms': x_range_ms, 'late_persistence_periods_ms': late_ms}

    def _hp_on_highlight_periods_ms(self, request: BaseRequest) -> list:
        """Union of intervals where LC or LA state name contains 'HpOn' (epoch ms)."""
        end_ms = int(self.data[request]['max_timestamp'].timestamp() * 1000)

        def segments_from_all(states_all: dict, order: list) -> list:
            pairs = sorted(zip(states_all['times'], states_all['values']))
            if not pairs:
                return []
            collapsed = []
            for t, idx in pairs:
                if collapsed and collapsed[-1][0] == t:
                    collapsed[-1] = (t, idx)
                else:
                    collapsed.append((t, idx))
            out = []
            for i, (t0, idx) in enumerate(collapsed):
                if idx < 0 or idx >= len(order):
                    continue
                if 'HpOn' not in order[idx]:
                    continue
                t0_ms = int(t0)
                t1_ms = int(collapsed[i + 1][0]) if i + 1 < len(collapsed) else end_ms
                if t1_ms > t0_ms:
                    out.append([t0_ms, t1_ms])
            return out

        lc = self.data[request].get('lc_states', {}).get('all', {'times': [], 'values': []})
        la = self.data[request].get('la_states', {}).get('all', {'times': [], 'values': []})
        raw = segments_from_all(lc, self.lc_states_order) + segments_from_all(
            la, self.la_states_order
        )
        if not raw:
            return []
        raw.sort(key=lambda p: p[0])
        merged = [raw[0][:]]
        for s, e in raw[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def _depth_scale_cutoff_ms(self) -> int:
        return int(pendulum.datetime(2026, 1, 9, tz=self.timezone_str).timestamp() * 1000)
        
    async def plot_heatpump(self, request: DataRequest):
        """Return raw series and axis metadata; styling and Plotly traces are built in the client."""
        plot_start = time.time()
        heatpump_channel_names = [
            'hp-lwt', 'hp-ewt', 'hp-odu-pwr', 'hp-idu-pwr',
            'oil-boiler-pwr', 'primary-flow', 'primary-pump-pwr',
        ]
        channels = {}
        for name in heatpump_channel_names:
            if name in self.data[request]['channels']:
                src = self.data[request]['channels'][name]
                channels[name] = {'times': src['times'], 'values': src['values']}
        print(f"Heat pump data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'heatpump',
            'channels': channels,
            'hp_on_highlight_periods_ms': self._hp_on_highlight_periods_ms(request),
            **self._plot_axis_meta(request),
        }

    async def plot_distribution(self, request: DataRequest):
        plot_start = time.time()
        names = ['dist-swt', 'dist-rwt', 'dist-flow', 'dist-pump-pwr']
        channels = {}
        for name in names:
            if name in self.data[request]['channels']:
                src = self.data[request]['channels'][name]
                channels[name] = {'times': src['times'], 'values': src['values']}
        print(f"Distribution data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {'plotKind': 'distribution', 'channels': channels, **self._plot_axis_meta(request)}
    
    async def plot_heatcalls(self, request: DataRequest):
        plot_start = time.time()
        threshold = self.whitewire_threshold_watts.get(
            request.house_alias, self.whitewire_threshold_watts['default']
        )
        zones = []
        for zone in self.data[request]['channels_by_zone'].keys():
            if 'whitewire' not in self.data[request]['channels_by_zone'][zone]:
                continue
            whitewire_ch = self.data[request]['channels_by_zone'][zone]['whitewire']
            ch_data = self.data[request]['channels'][whitewire_ch]
            zones.append({
                'zone_number': int(whitewire_ch[4]),
                'legend_name': whitewire_ch.replace('-whitewire', ''),
                'times': ch_data['times'],
                'values': ch_data['values'],
            })
        zone_axis_count = len(self.data[request]['channels_by_zone'].keys())
        print(f"Heat calls data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'heatcalls',
            'zones': zones,
            'zone_axis_count': zone_axis_count,
            'whitewire_threshold': threshold,
            'zone_colors': list(self.zone_color),
            **self._plot_axis_meta(request),
        }
    
    async def plot_zones(self, request: DataRequest):
        plot_start = time.time()
        zone_list = []
        for zone in self.data[request]['channels_by_zone']:
            zd = int(zone[4])
            entry = {'zone_key': zone, 'zone_digit': zd, 'temp': None, 'set': None}
            if 'temp' in self.data[request]['channels_by_zone'][zone]:
                tc = self.data[request]['channels_by_zone'][zone]['temp']
                src = self.data[request]['channels'][tc]
                entry['temp'] = {
                    'times': src['times'],
                    'values': src['values'],
                    'legend_suffix': tc.replace('-temp', ''),
                }
            if 'set' in self.data[request]['channels_by_zone'][zone]:
                sc = self.data[request]['channels_by_zone'][zone]['set']
                src = self.data[request]['channels'][sc]
                entry['set'] = {
                    'times': src['times'],
                    'values': src['values'],
                    'legend_suffix': sc.replace('-set', ''),
                }
            zone_list.append(entry)
        oat = None
        if 'oat' in self.data[request]['channels']:
            o = self.data[request]['channels']['oat']
            oat = {'times': o['times'], 'values': o['values']}
        print(f"Zones data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'zones',
            'zones': zone_list,
            'oat': oat,
            'zone_colors': list(self.zone_color),
            **self._plot_axis_meta(request),
        }
    
    async def plot_buffer(self, request: DataRequest):
        plot_start = time.time()
        cutoff = self._depth_scale_cutoff_ms()
        use_decicelsius = request.end_ms >= cutoff
        buffer_depths = []
        buffer_channels = sorted(
            key for key in self.data[request]['channels'].keys()
            if 'buffer-depth' in key and 'micro-v' not in key and 'device' not in key
        )
        for bc in buffer_channels:
            src = self.data[request]['channels'][bc]
            buffer_depths.append({'key': bc, 'times': src['times'], 'values': src['values']})
        hot = cold = None
        if 'buffer-hot-pipe' in self.data[request]['channels']:
            h = self.data[request]['channels']['buffer-hot-pipe']
            hot = {'times': h['times'], 'values': h['values']}
        if 'buffer-cold-pipe' in self.data[request]['channels']:
            c = self.data[request]['channels']['buffer-cold-pipe']
            cold = {'times': c['times'], 'values': c['values']}
        print(f"Buffer data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'buffer',
            'use_decicelsius_depth_scale': use_decicelsius,
            'buffer_depths': buffer_depths,
            'buffer_hot_pipe': hot,
            'buffer_cold_pipe': cold,
            **self._plot_axis_meta(request),
        }

    async def plot_storage(self, request: DataRequest):
        plot_start = time.time()
        cutoff = self._depth_scale_cutoff_ms()
        use_decicelsius = request.end_ms >= cutoff
        tank_depths = []
        seen_norm = set()
        tank_raw = sorted(
            key for key in self.data[request]['channels'].keys()
            if 'tank' in key and 'micro-v' not in key and 'device' not in key
        )
        for raw in tank_raw:
            norm = raw.split('depth')[0] + 'depth' + raw.split('depth')[1].split('-')[0]
            if norm in seen_norm or norm not in self.data[request]['channels']:
                continue
            seen_norm.add(norm)
            src = self.data[request]['channels'][norm]
            tank_depths.append({'key': norm, 'times': src['times'], 'values': src['values']})

        def _series(name):
            if name not in self.data[request]['channels']:
                return None
            s = self.data[request]['channels'][name]
            return {'times': s['times'], 'values': s['values']}

        print(f"Storage data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'storage',
            'use_decicelsius_depth_scale': use_decicelsius,
            'tank_depths': tank_depths,
            'store_hot_pipe': _series('store-hot-pipe'),
            'store_cold_pipe': _series('store-cold-pipe'),
            'store_pump_pwr': _series('store-pump-pwr'),
            'store_flow': _series('store-flow'),
            'usable_energy': _series('usable-energy'),
            'required_energy': _series('required-energy'),
            **self._plot_axis_meta(request),
        }
    
    async def plot_top_state(self, request: DataRequest):
        plot_start = time.time()
        top_states = {
            k: {'times': v['times'], 'values': v['values']}
            for k, v in self.data[request]['top_states'].items()
        }
        print(f"Top state data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'top_state',
            'top_states': top_states,
            **self._plot_axis_meta(request),
        }
    
    async def plot_ha_state(self, request: DataRequest):
        plot_start = time.time()
        lc_states = {
            k: {'times': v['times'], 'values': v['values']}
            for k, v in self.data[request]['lc_states'].items()
        }
        print(f"HA state data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'ha_state',
            'lc_states': lc_states,
            **self._plot_axis_meta(request),
        }
    
    async def plot_aa_state(self, request: DataRequest):
        plot_start = time.time()
        la_states = {
            k: {'times': v['times'], 'values': v['values']}
            for k, v in self.data[request]['la_states'].items()
        }
        print(f"AA state data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'aa_state',
            'la_states': la_states,
            **self._plot_axis_meta(request),
        }

    async def plot_weather(self, request: DataRequest):
        plot_start = time.time()
        oat_forecasts = {}
        for message in self.data[request]['weather_forecasts']:
            forecast_start_time = int((message.message_persisted_ms / 1000 // 3600) * 3600)
            oat_forecasts[forecast_start_time] = message.payload['OatF']
        runs = []
        n = len(oat_forecasts)
        for i, weather_time in enumerate(oat_forecasts):
            oat = oat_forecasts[weather_time]
            times_s = [int(weather_time) + 3600 * j for j in range(len(oat))]
            runs.append({
                'times_ms': [t * 1000 for t in times_s],
                'oat_f': list(oat),
                'run_index': i,
                'is_latest': i == n - 1,
            })
        print(f"Weather data payload done in {round(time.time() - plot_start, 1)} seconds")
        return {
            'plotKind': 'weather',
            'forecast_runs': runs,
            **self._plot_axis_meta(request),
        }
    
    async def plot_prices(self, request: Union[DataRequest, BaseRequest]):
        
        plot_start = time.time()

        # Query the external price API
        price_request = {
            "start_unix_s": request.start_ms/1000,
            "end_unix_s": request.end_ms/1000 + 48*3600,
            "timezone_str": "America/New_York"
        }
        
        try:
            price_service_start = time.time()
            print(f"Getting prices from price service...")
            async with httpx.AsyncClient() as client:
                response = await client.post("https://price-service.electricity.works/get_prices_visualizer/hw1-isone-me-versant-keene-ps/gw0-price-forecast", json=price_request)
                response.raise_for_status()
                data = response.json()
                lmp_values = data['LmpList']
                dist_values = data['DistList']
                print(f"Prices received from price service ({round(time.time()-price_service_start, 1)}s)")
        except Exception as e:
            print(f"Error getting prices from price service: {e}")
            lmp_values = []
            dist_values = []
        
        n = len(lmp_values)
        price_times_ms = [int(request.start_ms + i * 3600 * 1000) for i in range(n)]
        print(f"Prices data payload done in {round(time.time()-plot_start,1)} seconds")

        return {
            'plotKind': 'prices',
            'price_times_ms': price_times_ms,
            'lmp_values': lmp_values,
            'dist_values': dist_values,
            **self._plot_axis_meta(request),
        }

    async def get_homes(self, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
        print(f"Fetching homes for user: {current_user.username}")
        try:
            houses = db.execute(homes.select()).all()
            print(f"Found {len(houses)} houses")
            return houses
        except Exception as e:
            print(f"Error fetching houses: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching houses: {str(e)}")
        
    async def plot_electricity_use(self, hourly_data, request: ElectricityUseRequest):
        plot_start = time.time()

        white_color = '#858585' if request.darkmode else '#6c757d'

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[x+timedelta(minutes=30) for x in hourly_data['timestamps']],
                y=hourly_data['total_kwh'],
                opacity=0.6 if request.darkmode else 0.3,
                marker=dict(color='#2a4ca2', line=dict(width=0)),
                name='Electricity used',
                hovertemplate="%{x|%H}:00-%{x|%H}:59 | %{y:.1f} kWh<extra></extra>",
                width=[3600000/1.2] * len(hourly_data['timestamps']),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=hourly_data['timestamps'],
                y=hourly_data['prices'],
                mode='lines',
                opacity=0.8,
                showlegend=True,
                line_shape='hv',
                name='Electricity price',
                yaxis='y2',
                hovertemplate="%{x|%H:%M} | %{y:.2f} $/MWh<extra></extra>"
            )
        )

        hourly_data['prices'] = [x for x in hourly_data['prices'] if x is not None]
        hourly_data['total_kwh'] = [x for x in hourly_data['total_kwh'] if x is not None]
                
        fig.update_layout(
            title=dict(text='', x=0.5, xanchor='center'),
            plot_bgcolor='#1b1b1c' if request.darkmode else 'white',
            paper_bgcolor='#1b1b1c' if request.darkmode else 'white',
            font_color=white_color,
            title_font_color=white_color,
            margin=dict(t=30, b=30),
            xaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=white_color,
                showgrid=False,
            ),
            yaxis=dict(
                title='Quantity [kWh]',
                range = [0, (1.3*max(hourly_data['total_kwh']) if max(hourly_data['total_kwh']) > 3 else 1.3*3)],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=white_color,
                zeroline=False,
                showgrid=False,
                gridwidth=1,
                gridcolor=white_color
            ),
            yaxis2=dict(
                title='Price [$/MWh]',
                range = [
                    0 if not hourly_data['prices'] else min(hourly_data['prices'])-5, 
                    1.3*(10 if not hourly_data['prices'] else max(hourly_data['prices']))
                ],
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=False,
                linecolor=white_color,
                showgrid=False,
                overlaying='y',
                side='right'
            ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
            ),
        )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Electricity use plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
        
    async def get_electricity_use(self, request: ElectricityUseRequest, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
        if (
            isinstance(request.selected_short_aliases, list) 
            and len(request.selected_short_aliases) == 1 
            and isinstance(request.selected_short_aliases[0], str) 
            and ',' in request.selected_short_aliases[0]
        ):
            request.selected_short_aliases = [s.strip() for s in request.selected_short_aliases[0].split(',')]
        try:
            # Query the database for electricity records matching the selected short_aliases
            query = select(hourly_electricity).where(
                hourly_electricity.c.short_alias.in_(request.selected_short_aliases),
                hourly_electricity.c.hour_start_s >= request.start_ms // 1000,
                hourly_electricity.c.hour_start_s <= request.end_ms // 1000
            ).order_by(hourly_electricity.c.hour_start_s)
            
            records = db.execute(query).all()
            
            if not records:
                print(f"No electricity data found for the selected houses")
                return {"success": False}
                # raise HTTPException(status_code=404, detail="No electricity data found for the selected houses")
            
            # Group the data by timestamp and sum the kwh values
            timestamps = []
            total_kwh = []   
            prices = []
            for record in records:
                hour_start_s_rounded = (record.hour_start_s // 3600) * 3600
                if hour_start_s_rounded not in timestamps:
                    timestamps.append(hour_start_s_rounded)
                    total_kwh.append(record.hp_kwh_el)
                    if record.total_usd_per_mwh is not None:
                        prices.append(record.total_usd_per_mwh)
                    else:
                        date_time = pendulum.from_timestamp(hour_start_s_rounded, tz='America/New_York')
                        hour = date_time.hour
                        weekday = date_time.weekday()
                        dist_price = (
                            487.63 if hour in [7,8,9,10,11,16,17,18,19] and weekday<5 
                            else 54.98 if hour in [12,13,14,15] and weekday<5
                            else 50.13
                        )
                        prices.append(dist_price)
                else:
                    idx = timestamps.index(hour_start_s_rounded)
                    total_kwh[idx] += record.hp_kwh_el

            hourly_data = {
                "timestamps": [self.to_datetime(x*1000) for x in timestamps],
                "total_kwh": total_kwh,
                "prices": prices
            }

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                html_buffer = await self.plot_electricity_use(hourly_data, request)
                zip_file.writestr('plot1.html', html_buffer.read())
            zip_buffer.seek(0)

            return StreamingResponse(
                zip_buffer, 
                media_type='application/zip', 
                headers={"Content-Disposition": "attachment; filename=plots.zip"}
                )
            
        except Exception as e:
            print(f"Error getting electricity use: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting electricity use: {str(e)}")

    async def get_electricity_use_csv(self, request: ElectricityUseRequest, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
        if (
            isinstance(request.selected_short_aliases, list) 
            and len(request.selected_short_aliases) == 1 
            and isinstance(request.selected_short_aliases[0], str) 
            and ',' in request.selected_short_aliases[0]
        ):
            request.selected_short_aliases = [s.strip() for s in request.selected_short_aliases[0].split(',')]
        try:
            # Query the database for electricity records matching the selected short_aliases
            query = select(hourly_electricity).where(
                hourly_electricity.c.short_alias.in_(request.selected_short_aliases),
                hourly_electricity.c.hour_start_s >= request.start_ms // 1000,
                hourly_electricity.c.hour_start_s <= request.end_ms // 1000
            ).order_by(hourly_electricity.c.hour_start_s)
            
            records = db.execute(query).all()
            
            if not records:
                return {"success": False}
            
            # Group the data by timestamp and sum the kwh values
            timestamps = []
            total_kwh = []            
            for record in records:
                hour_start_s_rounded = (record.hour_start_s // 3600) * 3600
                if hour_start_s_rounded not in timestamps:
                    timestamps.append(hour_start_s_rounded)
                    total_kwh.append(record.hp_kwh_el)
                else:
                    idx = timestamps.index(hour_start_s_rounded)
                    total_kwh[idx] += record.hp_kwh_el

            # Convert timestamps to datetime objects in America/New_York timezone
            datetime_timestamps = []
            for ts in timestamps:
                dt = pd.to_datetime(ts*1000, unit='ms', utc=True)
                dt = dt.tz_convert('America/New_York').replace(tzinfo=None)
                datetime_timestamps.append(dt)

            # Create DataFrame
            if len(request.selected_short_aliases) > 1:
                df = pd.DataFrame({
                    'timestamp': datetime_timestamps,
                    'kwh': [round(x,2) for x in total_kwh]
                })
            else:
                df = pd.DataFrame([dict(row._mapping) for row in records])
                df['hour_start'] = pd.to_datetime(df['hour_start_s'] * 1000, unit='ms', utc=True)
                df['hour_start'] = df['hour_start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
                df = df.drop(columns=['hour_start_s', 'g_node_alias', 'short_alias'], errors='ignore')
            # Make the hour_start column the first column
            if 'hour_start' in df.columns:
                cols = df.columns.tolist()
                cols.insert(0, cols.pop(cols.index('hour_start')))
                df = df[cols]

            # Build file name
            start_date = self.to_datetime(request.start_ms)
            end_date = self.to_datetime(request.end_ms)
            formatted_start_date = start_date.strftime('%Y-%m-%d-%H-%M')
            formatted_end_date = end_date.strftime('%Y-%m-%d-%H-%M')
            house_alias = request.selected_short_aliases[0] if len(request.selected_short_aliases) == 1 else 'aggregated'
            filename = f'{house_alias}_electricity_use_{formatted_start_date}-{formatted_end_date}.csv'

            # Create CSV buffer
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            return StreamingResponse(
                iter([csv_buffer.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        except Exception as e:
            print(f"Error getting electricity use CSV: {e}")
            return {"success": False, "message": "An error occurred while getting electricity use CSV", "reload": False}

    async def update_scada_code(self, request: ScadaUpdateRequest, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
        print(f"Updating SCADA code for {request.selected_short_aliases}")

        results = []
        for house_alias in request.selected_short_aliases:
            print(f"Updating SCADA code for {house_alias}...")
            query = select(homes).where(homes.c.short_alias == house_alias)
            result = db.execute(query).first()
            if not result:
                results.append({
                    "house_alias": house_alias,
                    "status": "error",
                    "message": f"House with alias {house_alias} not found"
                })
                continue
            
            ssh_host = result.scada_ip_address
            if not ssh_host:
                results.append({
                    "house_alias": house_alias,
                    "status": "error",
                    "message": "SSH host information not found for this house"
                })
                continue
            
            try:
                # Step 1: Git operations (checkout, pull) and update database with commit hash
                print(f"Step 1: Updating git repository for {house_alias}...")
                git_ssh_command = f"ssh -o StrictHostKeyChecking=no -A pi@{ssh_host} 'cd ~/gridworks-scada && git checkout . && git checkout main && git pull'"
                git_process = await asyncio.create_subprocess_shell(
                    git_ssh_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                git_stdout, git_stderr = await git_process.communicate()
                
                if git_process.returncode != 0:
                    print(f"Failed to update git repository: {git_stderr.decode()}")
                    # Update the database with error status
                    try:
                        commit_hash = 'git_update_failed'
                        print(f"Updating database with commit hash: {commit_hash}")
                        update_query = homes.update().where(homes.c.short_alias == house_alias).values(scada_git_commit=commit_hash)
                        db.execute(update_query)
                        db.commit()
                    except Exception as db_error:
                        print(f"Error updating database with commit hash: {db_error}")
                    results.append({
                        "house_alias": house_alias,
                        "status": "error",
                        "message": f"Failed to update git repository: {git_stderr.decode()}"
                    })
                    continue
                
                # Get the git commit hash after successful pull
                git_info_command = f"ssh -o StrictHostKeyChecking=no pi@{ssh_host} 'cd ~/gridworks-scada && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD && git log -1 --pretty=format:\"%h - %s (%cr)\"'"
                git_info_process = await asyncio.create_subprocess_shell(
                    git_info_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                git_info_stdout, git_info_stderr = await git_info_process.communicate()

                if git_info_process.returncode != 0:
                    print(f"Failed to get git commit hash: {git_info_stderr.decode()}")
                    commit_info = "Failed to get commit information"
                    commit_hash_with_branch = "unknown"
                else:
                    commit_info = git_info_stdout.decode().strip()
                    commit_lines = commit_info.split('\n')
                    commit_hash = commit_lines[0][:7] if commit_lines else "unknown"
                    branch_name = commit_lines[1] if len(commit_lines) > 1 else "unknown"
                    commit_hash_with_branch = f"{branch_name} / {commit_hash}"
                
                # Update the database with the new commit hash
                try:
                    print(f"Updating database with commit hash: {commit_hash_with_branch}")
                    update_query = homes.update().where(homes.c.short_alias == house_alias).values(scada_git_commit=commit_hash_with_branch)
                    db.execute(update_query)
                    db.commit()
                except Exception as db_error:
                    print(f"Error updating database with commit hash: {db_error}")
                
                # Step 2: Service management (stop, update packages if needed, start)
                print(f"Step 2: Managing services for {house_alias}...")
                if request.update_packages:
                    service_ssh_command = f"ssh -o StrictHostKeyChecking=no -A pi@{ssh_host} 'cd ~/gridworks-scada && /home/pi/.local/bin/gwstop && ./tools/mkenv-pi.sh && /home/pi/.local/bin/gwstart'"
                else:
                    service_ssh_command = f"ssh -o StrictHostKeyChecking=no -A pi@{ssh_host} 'cd ~/gridworks-scada && /home/pi/.local/bin/gwstop && /home/pi/.local/bin/gwstart'"
                
                service_process = await asyncio.create_subprocess_shell(
                    service_ssh_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                service_stdout, service_stderr = await service_process.communicate()
                
                if service_process.returncode != 0:
                    print(f"Failed to manage services: {service_stderr.decode()}")
                    results.append({
                        "house_alias": house_alias,
                        "status": "partial_success",
                        "message": f"Git repository updated successfully, but service management failed: {service_stderr.decode()}",
                        "commit_info": commit_info
                    })
                else:
                    results.append({
                        "house_alias": house_alias,
                        "status": "success",
                        "message": "SCADA code updated and services restarted successfully",
                        "commit_info": commit_info
                    })
                
            except Exception as e:
                results.append({
                    "house_alias": house_alias,
                    "status": "error",
                    "message": f"Error updating SCADA code: {str(e)}"
                })
        
        return {
            "status": "completed",
            "results": results
        }

if __name__ == "__main__":
    a = VisualizerApi()
    a.start()
