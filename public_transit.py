import math
from matplotlib import colors
from util import *
import geohash
import matplotlib.cm as cmx
import matplotlib as mpl
import networkx as nx

mpl.rcParams['agg.path.chunksize'] = 10000

HASH_NODE_CSV = "Data/Geo/hash_node.csv"
GRAPH_TO_CSV = "Data/GoogleMaps/hcm_gmaps_df.csv"
FULL_GRAPH = "Data/Graphs/hcm_full_graph.gml"
WALKING_GRAPH = "Data/Graphs/hcm_walking_graph.gml"
TRANSIT_GRAPH = "Data/Graphs/hcm_transit_graph.gml"

CLUSTER_COLORS = ['r', 'g', 'b', 'c', 'm']


class PublicTransport(object):
    def __init__(self, graph_path=None):
        self.data = pd.read_csv(ZONE_INFO_PATH)
        self.hash_to_node = {}
        self.node_to_hash = {}
        self.create_mapping()
        self.node_count_when_building = max(self.data.id) + 1
        self.graph = self.load_graph(graph_path)
        self.get_hash2node_node2hash()
        print("Number of nodes: {}".format(self.graph.number_of_nodes()))
        print("Number of edges: {}".format(self.graph.number_of_edges()))

    def get_hash2node_node2hash(self):
        for i, row in pd.read_csv(HASH_NODE_CSV).iterrows():
            self.hash_to_node[row.hash_name] = row.node_id
            self.node_to_hash[str(row.node_id)] = row.hash_name

    def create_mapping(self):
        for i, row in self.data.iterrows():
            hash_name = geohash.encode(round(row.latitude, 3), round(row.longitude, 3))
            self.hash_to_node[hash_name] = row.id
            self.node_to_hash[str(row.id)] = hash_name

    def load_graph(self, graph_path):
        if graph_path:
            return nx.read_gml(graph_path)
        else:
            return nx.Graph()

    def save_graph(self, graph_path):
        nx.write_gml(self.graph, graph_path)

    def create_full_graph(self):
        """
        Compute Graph based on intermediate steps from the Google Response.
        """
        edge_id = 0
        self.count_dup = 0
        df = pd.read_pickle(GOOGLE_RESPONSE_DATA)
        for i, row in df.iterrows():
            try:
                directions = row.response[0]["legs"][0]
                self.intermediate_locations(directions)
            except:
                pass

        self.save_graph(FULL_GRAPH)
        d = {'node_id': self.hash_to_node.values(), 'hash_name': self.hash_to_node.keys()}
        df = pd.DataFrame(d)
        df.to_csv(HASH_NODE_CSV)
        print("Number of duplicate edges:", self.count_dup)
        print("The total number of Nodes is:", self.graph.number_of_nodes())
        print("The total number of Edges is:", self.graph.number_of_edges())

    def intermediate_locations(self, directions):
        """
        Gives the recursive directions based on some google response
        """
        locations = [direction['start_location'] for direction in directions["steps"]]
        locations.append(directions["steps"][-1]["end_location"])
        for i in range(len(locations) - 1):
            direction = directions["steps"][i]
            distance_meters = direction['distance']['value']
            duration_seconds = direction['duration']['value']
            travel_mode = direction['travel_mode']
            self.add_nodes_and_edges_full_graph(locations[i], locations[i + 1],
                                                distance_meters, duration_seconds, travel_mode=travel_mode)

    def add_nodes_and_edges_full_graph(self, zone1_id, zone2_id,
                                       distance_meters, duration_seconds, travel_mode=None):
        # Add nodes
        zone1_id_key = geohash.encode(round(zone1_id[u'lat'], 3), round(zone1_id[u'lng'], 3))
        zone2_id_key = geohash.encode(round(zone2_id[u'lat'], 3), round(zone2_id[u'lng'], 3))
        if zone1_id_key not in self.hash_to_node.keys():
            self.hash_to_node[zone1_id_key] = self.node_count_when_building
            self.node_to_hash[str(self.node_count_when_building)] = zone1_id_key
            self.node_count_when_building += 1
        if zone2_id_key not in self.hash_to_node.keys():
            self.hash_to_node[zone2_id_key] = self.node_count_when_building
            self.node_to_hash[str(self.node_count_when_building)] = zone2_id_key
            self.node_count_when_building += 1
        zone1_id = self.hash_to_node[zone1_id_key]
        zone2_id = self.hash_to_node[zone2_id_key]

        if not self.graph.has_node(zone1_id):
            self.graph.add_node(zone1_id)
        if not self.graph.has_node(zone2_id):
            self.graph.add_node(zone2_id)
        if not self.graph.has_edge(zone1_id, zone2_id):
            self.graph.add_edge(zone1_id, zone2_id,
                                distance_meters=distance_meters,
                                duration_seconds=duration_seconds,
                                weight=1, travel_mode=travel_mode)
        else:
            # Increment duplicate count
            self.count_dup += 1
            # Add to weight of edge(s)
            prev_weight = self.graph.get_edge_data(zone1_id, zone2_id)['weight']
            self.graph.add_edge(zone1_id, zone2_id, weight=prev_weight + 1)

    def create_subgraphs(self):
        """
        Build two subgraphs based on the graph plus intermediate steps
        There are two graphs, either 'WALKING', 'TRANSIT'.
        """
        walking_graph = nx.Graph()
        transit_system_graph = nx.Graph()
        for edge in self.graph.edges(data=True):
            edge_data = self.graph.get_edge_data(edge[0], edge[1])
            distance_meters = edge_data["distance_meters"]
            duration_seconds = edge_data["duration_seconds"]
            travel_mode = edge_data["travel_mode"]
            edge_weight = edge_data['weight']

            if travel_mode == "WALKING":
                self.add_nodes_and_edges(walking_graph, edge[0], edge[1], distance_meters, duration_seconds,
                                         travel_mode, edge_weight)
            elif travel_mode == "TRANSIT":
                self.add_nodes_and_edges(transit_system_graph, edge[0], edge[1], distance_meters,
                                         duration_seconds, travel_mode, edge_weight)

        # Walking
        num_edges = walking_graph.number_of_edges()
        num_nodes = walking_graph.number_of_nodes()
        print('Number of nodes for the walking graph is: {}'.format(num_nodes))
        print('Number of edges for the walking graph is: {}'.format(num_edges))

        # Transit
        num_edges = transit_system_graph.number_of_edges()
        num_nodes = transit_system_graph.number_of_nodes()
        print('Number of nodes for the transit system graph is: {}'.format(num_nodes))
        print('Number of edges for the transit system graph is: {}'.format(num_edges))

        save_graph(walking_graph, WALKING_GRAPH)
        save_graph(transit_system_graph, TRANSIT_GRAPH)

    def add_nodes_and_edges(self, graph, node1, node2, distance_meters, duration_seconds,
                            travel_mode, edge_weight):
        if not graph.has_node(node1): graph.add_node(node1)
        if not graph.has_node(node2): graph.add_node(node2)
        if not graph.has_edge(node1, node2):
            graph.add_edge(node1, node2,
                           distance_meters=distance_meters,
                           duration_seconds=duration_seconds,
                           weight=edge_weight,
                           travel_mode=travel_mode
                           )

    def save_graph_to_csv(self):
        distances_m = []
        times_s = []
        travel_modes = []
        weights = []
        origin_nodes = []
        destination_nodes = []
        for edge in self.graph.edges(data=True):
            node1 = edge[0]
            node2 = edge[1]
            distance_m = self.graph.get_edge_data(node1, node2)["distance_meters"]
            time_s = self.graph.get_edge_data(node1, node2)["duration_seconds"]
            travel_mode = self.graph.get_edge_data(node1, node2)["travel_mode"]
            weight = self.graph.get_edge_data(node1, node2)["weight"]
            distances_m.append(distance_m)
            times_s.append(time_s)
            travel_modes.append(travel_mode)
            weights.append(weight)
            origin_nodes.append(node1)
            destination_nodes.append(node2)
        d = {"origin": origin_nodes, "destination": destination_nodes, "distance meters": distances_m,
             "time sec": times_s, "weight": weights, "travel mode": travel_modes}
        df = pd.DataFrame(d)
        df.to_csv(GRAPH_TO_CSV)

    def draw_map(self, filename, plot_edges=False, edge_weight_threshold=None, edge_scaling=None,
                 plot_nodes=False, node_scaling=None, centrality=None, classification=None):

        ###################################################
        # Always the same
        ###################################################
        # Extract polygons
        polys = MultiPolygon([shape(zone['geometry']) for zone in fiona.open(SHAPE_PATH)])
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

        ###################################################
        # Plot edges
        ###################################################
        if plot_edges:
            X, Y, scaling = [], [], []
            for i, edge in tqdm(enumerate(self.graph.edges(data=True))):
                node1 = edge[0]
                node2 = edge[1]
                start = geohash.decode(self.node_to_hash[str(node1)])
                end = geohash.decode(self.node_to_hash[str(node2)])
                # Determine if edge should be added
                add = True
                if edge_weight_threshold:
                    if self.graph.get_edge_data(node1, node2)['weight'] <= edge_weight_threshold:
                        add = False
                # If edge should be added
                if add:
                    X.append((start[1], end[1]))
                    Y.append((start[0], end[0]))
                    if edge_scaling == 'weight':
                        scaling.append(self.graph.get_edge_data(node1, node2)[edge_scaling])
                    elif edge_scaling is not None:
                        scaling.append(self.graph.get_edge_data(node1, node2)[edge_scaling])
            # Plot
            if edge_scaling is None:
                ax.plot(X, Y, color='g', linewidth='1')
            else:
                print(edge_scaling.upper(), "min: {}, max: {}".format(min(scaling), max(scaling)))
                scaling = [math.log(x) if not int(x) == 0 else 0 for x in
                           scaling]  # Need to use log scaling when doing weights
                cmap = plt.get_cmap('YlOrRd')
                cNorm = colors.Normalize(vmin=min(scaling), vmax=max(scaling))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
                c = [scalarMap.to_rgba(x) for x in scaling]
                for i, x in enumerate(X):
                    ax.plot(X[i], Y[i], color=c[i])

        ###################################################
        # Plot nodes
        ###################################################
        if plot_nodes:
            lats, longs, degrees = [], [], []
            for node in self.graph.nodes():
                # Get lat and long of node
                long, lat = geohash.decode(self.node_to_hash[node])
                lats.append(lat)
                longs.append(long)
                # Use node degree
                degree = 0
                if node_scaling == 'degree':
                    degree = self.graph.degree(node)
                # Use weighted node degree
                elif node_scaling == 'degree_weighted':
                    for neighbor in self.graph.neighbors(node):
                        weight = self.graph.get_edge_data(node, neighbor)['weight']
                        degree += weight
                # Use node centrality
                elif node_scaling == 'centrality':
                    degree = centrality[int(node)]
                # Use node classification
                elif node_scaling == 'classification':
                    degree = classification[int(node)]
                # Append to degrees list
                degrees.append(degree)

            if not node_scaling == 'classification' and node_scaling is not None:
                # Scale degrees so that dots are properly sized
                max_degree = float(max(degrees))
                print(node_scaling.upper(), "min: {}, max: {}".format(min(degrees), max(degrees)))
                scales = [float(x) / max_degree * 300 for x in degrees]
                df = pd.DataFrame({"lats": lats, "longs": longs, "scales": scales, "degrees": degrees})
                df = df.sort_values(['degrees'])
                vis = ax.scatter(df.lats, df.longs, s=df.scales, c=df.degrees, cmap=plt.cm.get_cmap('plasma'))
                fig.colorbar(vis)
            elif node_scaling is not None:
                for i, lat in enumerate(lats):
                    ax.scatter(lats[i], longs[i], c=CLUSTER_COLORS[degrees[i]], s=30)
            else:
                for i, lat in enumerate(lats):
                    ax.scatter(lats[i], longs[i], color='r', s=30)

        ###################################################
        # Always the same
        ###################################################
        plt.box(False)
        ax.axis('off')
        plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    if False:
        public_transport = PublicTransport()
        public_transport.create_full_graph()
        public_transport.save_graph_to_csv()

    if False:
        public_transport = PublicTransport(FULL_GRAPH)
        public_transport.draw_map("Data/Images/Nodes/hcm_full_graph_nodes_degree.png", plot_nodes=True,
                                  node_scaling='degree')
        public_transport.draw_map("Data/Images/Nodes/hcm_full_graph_nodes_degree_weighted.png", plot_nodes=True,
                                  node_scaling='degree_weighted')
        public_transport.draw_map("Data/Images/Edges/hcm_full_graph_edges.png", plot_edges=True)
        public_transport.draw_map("Data/Images/Edges/hcm_full_graph_edges_weight.png", plot_edges=True,
                                  edge_scaling='weight')
        public_transport.draw_map("Data/Images/Edges/hcm_full_graph_edges_duration.png", plot_edges=True,
                                  edge_scaling='duration_seconds')
        public_transport.draw_map("Data/Images/Edges/hcm_full_graph_edges_distance.png", plot_edges=True,
                                  edge_scaling='distance_meters')

    if False:
        public_transport = PublicTransport(FULL_GRAPH)
        public_transport.create_subgraphs()

    if False:
        walking_graph = PublicTransport(WALKING_GRAPH)
        walking_graph.draw_map("Data/Images/Nodes/hcm_walking_graph_nodes_degree.png", plot_nodes=True,
                               node_scaling='degree')
        walking_graph.draw_map("Data/Images/Nodes/hcm_walking_graph_nodes_degree_weighted.png", plot_nodes=True,
                               node_scaling='degree_weighted')
        walking_graph.draw_map("Data/Images/Edges/hcm_walking_graph_edges.png", plot_edges=True)
        walking_graph.draw_map("Data/Images/Edges/hcm_walking_graph_edges_weight.png", plot_edges=True,
                               edge_scaling='weight')
        walking_graph.draw_map("Data/Images/Edges/hcm_walking_graph_edges_duration.png", plot_edges=True,
                               edge_scaling='duration_seconds')
        walking_graph.draw_map("Data/Images/Edges/hcm_walking_graph_edges_distance.png", plot_edges=True,
                               edge_scaling='distance_meters')

    if True:
        transit_graph = PublicTransport(TRANSIT_GRAPH)
        transit_graph.draw_map("Data/Images/Nodes/hcm_transit_graph_nodes_degree.png", plot_nodes=True,
                                  node_scaling='degree')
        transit_graph.draw_map("Data/Images/Nodes/hcm_transit_graph_nodes_degree_weighted.png", plot_nodes=True,
                                  node_scaling='degree_weighted')
        transit_graph.draw_map("Data/Images/Edges/hcm_transit_graph_edges.png", plot_edges=True)
        transit_graph.draw_map("Data/Images/Edges/hcm_transit_graph_edges_weight.png", plot_edges=True,
                                  edge_scaling='weight')
        transit_graph.draw_map("Data/Images/Edges/hcm_transit_graph_edges_duration.png", plot_edges=True,
                                  edge_scaling='duration_seconds')
        transit_graph.draw_map("Data/Images/Edges/hcm_transit_graph_edges_distance.png", plot_edges=True,
                                  edge_scaling='distance_meters')
