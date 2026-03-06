import csv
import gc
import os
import shutil
import dotenv
import pendulum
from sqlalchemy import asc, cast
from sqlalchemy import create_engine, select, BigInteger, MetaData, Table
from sqlalchemy.orm import sessionmaker
from visualizer.config import Settings
from visualizer.models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo.dijkstra_types import DNode, DEdge
from gridflo import Flo, DGraphVisualizer, DNodeVisualizer
from gridflo.asl.types import WinterOakSupergraphParams
from gridflo.supergraph_generator import SupergraphGenerator, RuleBasedStorageModel
from gridflo.supergraph_generator import WinterOakSupergraphParams
from gridflo.dijkstra_types import DNode
from gridflo.dgraph_visualizer import DNodeComparator
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image

# ---------------------------------------------------
# Gather user inputs
# ---------------------------------------------------

print("\nWelcome to the FLO analysis tool!\n")
house_alias = input("Enter house alias (default: oak): ")
if house_alias == "":
    house_alias = "oak"

now = pendulum.now(tz='America/New_York')
yesterday_8pm = now.subtract(days=1).set(hour=20, minute=0, second=0, microsecond=0)

start_input = input("Enter start year, month, day, hour (default: yesterday 8pm): ")
if start_input and len(start_input.split(',')) == 4:
    START_YEAR, START_MONTH, START_DAY, START_HOUR = start_input.split(',')
    START_YEAR, START_MONTH, START_DAY, START_HOUR = int(START_YEAR), int(START_MONTH), int(START_DAY), int(START_HOUR)
else:
    START_YEAR, START_MONTH, START_DAY, START_HOUR = yesterday_8pm.year, yesterday_8pm.month, yesterday_8pm.day, yesterday_8pm.hour

end_input = input("Enter end year, month, day, hour (default: now): ")
if end_input and len(end_input.split(',')) == 4:
    END_YEAR, END_MONTH, END_DAY, END_HOUR = end_input.split(',')
    END_YEAR, END_MONTH, END_DAY, END_HOUR = int(END_YEAR), int(END_MONTH), int(END_DAY), int(END_HOUR)
else:
    END_YEAR, END_MONTH, END_DAY, END_HOUR = now.year, now.month, now.day, now.hour

start_time = pendulum.datetime(START_YEAR, START_MONTH, START_DAY, START_HOUR, tz='America/New_York')
end_time = pendulum.datetime(END_YEAR, END_MONTH, END_DAY, END_HOUR, tz='America/New_York')

# TEMPORARY
house_alias = "oak"
start_time = pendulum.datetime(2026, 2, 21, 0, tz='America/New_York')
end_time = pendulum.datetime(2026, 2, 21, 10, tz='America/New_York')

start_ms = start_time.timestamp()*1000
end_ms = end_time.timestamp()*1000

print(f"Analyzing FLO report for {house_alias} from {start_time} to {end_time}\n")

# ---------------------------------------------------
# Load FLO report CSV file
# ---------------------------------------------------

csv_path = os.path.expanduser(f'~/Desktop/flo_report_{house_alias}.csv')
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}")
    print("Please run flo_report.py first to generate the report.")
    exit(1)

all_rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_rows.append(row)

start_str = start_time.format('YYYY-MM-DD HH:00')
end_str = end_time.format('YYYY-MM-DD HH:00')
rows = [row for row in all_rows if start_str <= row['Timestamp'] < end_str]
heat_to_house_true = [float(row['H2H_A']) for row in rows]

print(f"Loaded {len(rows)} of {len(all_rows)} rows from {csv_path} (filtered to {start_str} .. {end_str})")

# ---------------------------------------------------
# Find FLO params messages
# ---------------------------------------------------

stmt = select(MessageSql).filter(
    MessageSql.message_type_name == "flo.params.house0",
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(int(end_ms+10*60*1000), BigInteger),
    MessageSql.message_persisted_ms >= cast(int(start_ms-10*60*1000), BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
result = session.execute(stmt)
messages = result.scalars().all()

flo_params_messages: list[MessageSql] = []
for m in messages:
    if pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York').minute == 57:
        print(f"Adding message from {m.from_alias} at {pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York')}")
        flo_params_messages.append(m)

print(f"Found {len(messages)} messages and {len(flo_params_messages)} of them at minute 57\n")

session.close()
engine.dispose()

# ---------------------------------------------------
# Initialize the model from the parameters
# ---------------------------------------------------

flo_params = FloParamsHouse0(**flo_params_messages[0].payload)

supergraph_params = WinterOakSupergraphParams(
    num_layers=flo_params.num_layers,
    storage_volume_gallons=flo_params.storage_volume_gallons,
    constant_delta_t=flo_params.constant_delta_t,
    max_hp_kwh_th=25,
    max_load_kwh_th=20,
)

generator = SupergraphGenerator(supergraph_params)
generator.create_nodes()

model = RuleBasedStorageModel(
    supergraph_params,
    generator.nodes,
    generator.nodes_by,
    generator.logger
)

# ---------------------------------------------------
# Find the initial state
# ---------------------------------------------------

start_node = DNode(
    top_temp=flo_params.initial_top_temp_f,
    middle_temp=flo_params.initial_middle_temp_f,
    bottom_temp=flo_params.initial_bottom_temp_f,
    thermocline1=flo_params.initial_thermocline_1,
    thermocline2=flo_params.initial_thermocline_2,
    parameters=supergraph_params,
)
print(f"Initial state: {start_node}")

# ---------------------------------------------------
# Simulate each hour
# ---------------------------------------------------

if os.path.exists('plots'):
    shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

previous_estimate_storage_kwh_now = 0
previous_plan_hp_kwh_el_list = []

for i in range(len(rows)):
    print(f"\nProcessing hour {i+1}/{len(rows)}...")

    # Take the real message's payload (weather, prices, house params, etc.)
    # but override the initial storage state with our simulated state
    # and the stability penalty inputs with the previous hour's values
    payload = dict(flo_params_messages[i].payload)
    payload['InitialTopTempF'] = int(round(start_node.top_temp))
    payload['InitialMiddleTempF'] = int(round(start_node.middle_temp))
    payload['InitialBottomTempF'] = int(round(start_node.bottom_temp))
    payload['InitialThermocline1'] = int(round(start_node.thermocline1))
    payload['InitialThermocline2'] = int(round(start_node.thermocline2))
    payload['PreviousEstimateStorageKwhNow'] = previous_estimate_storage_kwh_now
    payload['PreviousPlanHpKwhElList'] = previous_plan_hp_kwh_el_list
    payload['StabilityPenaltyWeight'] = 0
    payload['StabilityPenaltyDecay'] = 0.9
    payload['StabilityPenaltyThresholdKwh'] = 10.0
    payload['StabilityPenaltyHorizonHours'] = 40
    flo_params = FloParamsHouse0(**payload)

    # Find the FLO's decision at this node
    g = Flo(flo_params.to_bytes())
    g.solve_dijkstra()
    g.generate_recommendation(flo_params.to_bytes())

    # Use the first price in the forecast as the clearing price
    clearing_price = flo_params.total_price_forecast[0]
    g.get_next_node_at_price(clearing_price)

    initial_node_edge: DEdge = [e for e in g.bid_edges[g.initial_node] if e.head == g.initial_node.next_node][0]
    heat_from_hp_expected = initial_node_edge.hp_heat_out
    heat_to_buffer_expected = 0

    # Save the graph plot
    v = DGraphVisualizer(g)
    v.plot(show=False, save_as=f'plots/flo{i+1}_graph.png')

    # Simulate the outcome
    heat_to_store = heat_from_hp_expected - heat_to_house_true[i] - heat_to_buffer_expected
    start_node = model.next_node(start_node, heat_to_store)

    previous_estimate_storage_kwh_now = g.initial_node.next_node.energy
    previous_plan_hp_kwh_el_list = g.initial_node.next_node.shortest_path_hp_kwh_el

    del g, v
    gc.collect()

# ---------------------------------------------------
# Generate PDF with graph plots (4 per page)
# ---------------------------------------------------

pdf_path = os.path.expanduser(f'~/Desktop/flo_analysis_{house_alias}.pdf')
if os.path.exists(pdf_path):
    os.remove(pdf_path)

print("\nGenerating PDF report...")
c = canvas.Canvas(pdf_path, pagesize=A4)
page_width, page_height = A4
margin = 0.5 * inch
graphs_per_page = 4
usable_height = page_height - 2 * margin
graph_height = (usable_height - (graphs_per_page - 1) * 0.2 * inch) / graphs_per_page
graph_width = page_width - 2 * margin
num_graphs = len(rows)

for page_start in range(0, num_graphs, graphs_per_page):
    if page_start > 0:
        c.showPage()

    for slot in range(graphs_per_page):
        graph_num = page_start + slot + 1
        if graph_num > num_graphs:
            break

        graph_path = f'plots/flo{graph_num}_graph.png'
        if not os.path.exists(graph_path):
            continue

        img = Image.open(graph_path)
        img_w, img_h = img.size
        aspect = img_h / img_w

        display_w = graph_width
        display_h = min(graph_height, display_w * aspect)
        display_w = min(graph_width, display_h / aspect)

        y = page_height - margin - (slot + 1) * graph_height - slot * 0.2 * inch
        c.drawImage(graph_path, margin, y, width=display_w, height=display_h)

c.save()
print(f"PDF saved as {pdf_path}")

if os.path.exists('plots'):
    shutil.rmtree('plots')
