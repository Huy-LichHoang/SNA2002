import fiona
import csv
import googlemaps
from datetime import datetime
import networkx as nx
import itertools
from tqdm import tqdm
import pandas as pd
from math import sin, cos, radians, asin, sqrt
from shapely.geometry import shape, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

GEO_PATH = "Data/Geo/hcm_geoboundaries.json"
SHAPE_PATH = "Data/Geo/hcm_geoboundaries.shp"
ZONE_INFO_PATH = "Data/Geo/hcm_zone_info.csv"
DISTANCE_GRAPH = "Data/Graphs/hcm_distance_graph.gml"
GOOGLE_RESPONSE_DATA = "Data/GoogleMaps/gmaps_response_data.pkl"

# GOOGLE_API_KEY = "AIzaSyC__jxYlTQMaAWLm0bCvWjFY8qlhP8lIeo"
GOOGLE_API_KEY = "AIzaSyDWDIdHnn37uFkE8a7BwZR9h2iPkPrq1wA"


def save_zone_info():
    """
    Hàm lưu thông tin tên và tọa độ tâm của một vùng
    """
    # Extract data
    data = fiona.open(SHAPE_PATH, encoding='utf-8')
    properties = [(zone['properties']['id'], zone['properties']['name']) for zone in data]
    polys = MultiPolygon([shape(zone['geometry']) for zone in data])
    centroids = [x.centroid for x in polys]
    # Save to csv
    with open(ZONE_INFO_PATH, 'w', newline='', encoding='utf-8') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id', 'address', 'longitude', 'latitude'])
        for i, row in enumerate(properties):
            to_write = list(row) + [centroids[i].x, centroids[i].y]
            csv_out.writerow(to_write)


def save_shp(data):
    """
    Hàm chuyển đổi dữ liệu từ file geo json sang file .shp
    """
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {'id': 'int', 'name': 'str'},
    }
    with fiona.open(SHAPE_PATH, 'w', 'ESRI Shapefile', schema, encoding='utf-8') as f:
        for zone in data:
            # id 2xx là Huyện Cần Giờ, id 3xx là Huyện Củ Chi
            if 200 < zone['properties']['id'] < 400 or zone['properties']['id'] in {2, 3}:
                continue
            f.write({'geometry': zone['geometry'],
                     'properties': {'id': zone['properties']['id'],
                                    'name': zone['properties']['name']},
                     })


def draw_map(image_name, district_border=False, plot_node=False):
    """
    Hàm vẽ bản đồ
    """
    # Extract polygons
    data = fiona.open(SHAPE_PATH, encoding='utf-8')
    if district_border:
        polys = MultiPolygon([shape(zone['geometry']) for zone in data])
    else:
        polys = MultiPolygon([shape(zone['geometry']) for zone in data if zone['properties']['id'] > 100])

    # Setup plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    min_x, min_y, max_x, max_y = polys.bounds
    w, h = max_x - min_x, max_y - min_y
    ax.set_xlim(min_x - 0.1 * w, max_x + 0.1 * w)
    ax.set_ylim(min_y - 0.1 * h, max_y + 0.1 * h)
    ax.set_aspect(1)

    # Plot zones
    patches = []
    for _, p in enumerate(polys):
        patches.append(PolygonPatch(p, fc='#AEEDFF', ec='#555555', alpha=1., zorder=1))
    ax.add_collection(PatchCollection(patches, match_original=True))

    if plot_node:
        with open(ZONE_INFO_PATH, encoding='utf-8') as f:
            csv_in = csv.reader(f)
            next(csv_in)
            for _, zone in enumerate(csv_in):
                if int(zone[0]) > 100:
                    ax.scatter(float(zone[2]), float(zone[3]), color='r', s=50)

    plt.box(False)
    ax.axis('off')
    # Save image
    plt.savefig(image_name, dpi=300)


def load_graph(file_path):
    return nx.read_gml(file_path)


def save_graph(graph, file_path):
    nx.write_gml(graph, file_path)


def calculate_distance(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Apply formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def create_distance_graph(data):
    # Initialize new multi graph
    graph = nx.Graph()

    # Add nodes
    for zone in data:
        if zone['properties']['id'] > 100:
            centroid = shape(zone['geometry']).centroid
            graph.add_node(zone['properties']['id'], lat_long=str(centroid.y) + ", " + str(centroid.x))
    num_nodes = graph.number_of_nodes()

    # Add edges
    prev, count = 0, 1
    for zone1, zone2 in itertools.combinations(data, 2):
        if zone1['properties']['id'] < 100 or zone2['properties']['id'] < 100:
            continue
        # Just for tracking progress
        cur = zone1['properties']['id']
        if not cur == prev:
            print('[Distance graph] Finished checking {} of {} zones'.format(count, num_nodes))
            prev = cur
            count += 1
        # The important stuff
        zone1_id, zone2_id = zone1['properties']['id'], zone2['properties']['id']
        zone1_centroid, zone2_centroid = shape(zone1['geometry']).centroid, shape(zone2['geometry']).centroid
        distance = calculate_distance(zone1_centroid.x, zone1_centroid.y, zone2_centroid.x, zone2_centroid.y)

        if not graph.has_edge(zone1_id, zone2_id):
            graph.add_edge(zone1_id, zone2_id, distance=distance)

    num_edges = graph.number_of_edges()

    # Print some properties of the graph
    print('Number of nodes (zones): {}'.format(num_nodes))
    print('Number of edges (zone borders): {}'.format(num_edges))

    # Save graph
    save_graph(graph, DISTANCE_GRAPH)


def request_google_maps(graph: nx.Graph):
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    now = datetime.now()  # 13-01-2021 15h30
    rows1 = []
    rows2 = []
    directions = []
    for count, edge in tqdm(enumerate(graph.edges(data=True))):
        try:
            row_1 = graph.nodes(data=True)[edge[0]]['lat_long']
            row_2 = graph.nodes(data=True)[edge[1]]['lat_long']
            directions_result = gmaps.directions(row_1, row_2, mode="transit", departure_time=now)
            rows1.append(row_1)
            rows2.append(row_2)
            directions.append(directions_result)

            if (count % 1000) == 0:
                d = {'node1': rows1, 'node2': rows2, "response": directions}
                df = pd.DataFrame(data=d)
                df.to_pickle("./dummy.pkl")
                print(count)

        except Exception as ex:
            print(ex)
            continue

    d = {'node1': rows1, 'node2': rows2, "response": directions}
    df = pd.DataFrame(d)
    df.to_pickle(GOOGLE_RESPONSE_DATA)
    print("Done")


if __name__ == '__main__':
    if False:
        with open(GEO_PATH, encoding='utf-8') as f:
            data = json.load(f)
            save_shp(data["features"])
            save_zone_info()

    if False:
        draw_map("Data/Images/Layouts/hcm_district_border.png", district_border=True, plot_node=True)
        draw_map("Data/Images/Layouts/hcm_nodes.png", district_border=False, plot_node=True)

    if False:
        data = fiona.open(SHAPE_PATH, encoding='utf-8')
        create_distance_graph(data)

    if False:
        graph = load_graph(DISTANCE_GRAPH)
        request_google_maps(graph)

