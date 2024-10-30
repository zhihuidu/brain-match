#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ITER 10000  // Maximum number of iterations for refinement

typedef struct {
    int target;
    int weight;
} AdjEdge;

typedef struct {
    AdjEdge *edges;
    int degree;
    int capacity;
} AdjacencyList;

typedef struct {
    AdjacencyList *adj;
    int num_nodes;
} Graph;

typedef struct {
    int *node_ids;  // Maps external node IDs to internal IDs 0 to n-1
    int count;
    int capacity;
} NodeMap;

// Initialize NodeMap
NodeMap *create_node_map(int initial_capacity) {
    NodeMap *map = malloc(sizeof(NodeMap));
    map->node_ids = malloc(initial_capacity * sizeof(int));
    map->count = 0;
    map->capacity = initial_capacity;
    return map;
}

// Find or add a node ID and return its mapped index
int add_node(NodeMap *map, int node_id) {
    for (int i = 0; i < map->count; i++) {
        if (map->node_ids[i] == node_id) {
            return i;
        }
    }

    if (map->count >= map->capacity) {
        map->capacity *= 2;
        map->node_ids = realloc(map->node_ids, map->capacity * sizeof(int));
    }

    map->node_ids[map->count] = node_id;
    return map->count++;
}

// Free NodeMap
void free_node_map(NodeMap *map) {
    free(map->node_ids);
    free(map);
}

// Initialize Graph
Graph *create_graph(int num_nodes) {
    Graph *graph = malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->adj = malloc(num_nodes * sizeof(AdjacencyList));
    for (int i = 0; i < num_nodes; i++) {
        graph->adj[i].edges = malloc(2 * sizeof(AdjEdge));  // Initial capacity of 2 edges
        graph->adj[i].degree = 0;
        graph->adj[i].capacity = 2;
    }
    return graph;
}

// Add edge to adjacency list
void add_edge(Graph *graph, int src, int tgt, int weight) {
    if (graph->adj[src].degree >= graph->adj[src].capacity) {
        graph->adj[src].capacity *= 2;
        graph->adj[src].edges = realloc(graph->adj[src].edges, graph->adj[src].capacity * sizeof(AdjEdge));
    }
    graph->adj[src].edges[graph->adj[src].degree++] = (AdjEdge){tgt, weight};
}

// Free Graph
void free_graph(Graph *graph) {
    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->adj[i].edges);
    }
    free(graph->adj);
    free(graph);
}

// Load graph from CSV and build adjacency list
Graph *load_graph(const char *filename, NodeMap *map) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    char line[256];
    fgets(line, sizeof(line), file);  // Skip header

    // Initialize graph (size will be set after reading all nodes)
    Graph *graph = NULL;

    while (fgets(line, sizeof(line), file)) {
        int src_id, tgt_id, weight;
        if (sscanf(line, "%d,%d,%d", &src_id, &tgt_id, &weight) != 3) {
            fprintf(stderr, "Error reading line: %s\n", line);
            continue;
        }

        int src = add_node(map, src_id);
        int tgt = add_node(map, tgt_id);

        // Initialize graph only after determining the number of unique nodes
        if (!graph) {
            graph = create_graph(map->capacity);
        }

        add_edge(graph, src, tgt, weight);
    }

    fclose(file);
    return graph;
}

// Load initial matching from a CSV file
int *load_initial_matching(const char *filename, NodeMap *map_g1, NodeMap *map_g2) {
    int *matching = malloc(map_g1->count * sizeof(int));
    for (int i = 0; i < map_g1->count; i++) {
        matching[i] = -1;  // Initialize with -1 (no match)
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening initial matching file: %s\n", filename);
        free(matching);
        return NULL;
    }

    char line[256];
    fgets(line, sizeof(line), file);  // Skip header

    while (fgets(line, sizeof(line), file)) {
        int node_g1_id, node_g2_id;
        if (sscanf(line, "%d,%d", &node_g1_id, &node_g2_id) != 2) {
            fprintf(stderr, "Error reading line: %s\n", line);
            continue;
        }

        int mapped_g1 = add_node(map_g1, node_g1_id);
        int mapped_g2 = add_node(map_g2, node_g2_id);
        matching[mapped_g1] = mapped_g2;
    }

    fclose(file);
    return matching;
}

// Calculate score change for a swap by looking only at edges connected to node1 and node2
int calculate_score_change(Graph *g1, Graph *g2, int *matching, int node1, int node2) {
    int delta_score = 0;

    // Check edges involving node1
    for (int i = 0; i < g1->adj[node1].degree; i++) {
        int target1 = g1->adj[node1].edges[i].target;
        int weight1 = g1->adj[node1].edges[i].weight;
        int target2 = matching[target1];

        int found = 0;
        for (int j = 0; j < g2->adj[matching[node1]].degree; j++) {
            if (g2->adj[matching[node1]].edges[j].target == target2) {
                delta_score += abs(weight1 - g2->adj[matching[node1]].edges[j].weight);
                found = 1;
                break;
            }
        }
        if (!found) {
            delta_score += weight1;
        }
    }

    // Check edges involving node2
    for (int i = 0; i < g1->adj[node2].degree; i++) {
        int target1 = g1->adj[node2].edges[i].target;
        int weight1 = g1->adj[node2].edges[i].weight;
        int target2 = matching[target1];

        int found = 0;
        for (int j = 0; j < g2->adj[matching[node2]].degree; j++) {
            if (g2->adj[matching[node2]].edges[j].target == target2) {
                delta_score += abs(weight1 - g2->adj[matching[node2]].edges[j].weight);
                found = 1;
                break;
            }
        }
        if (!found) {
            delta_score += weight1;
        }
    }

    return delta_score;
}

// Refine matching by performing random swaps
void refine_matching(Graph *g1, Graph *g2, int *matching) {
    int num_nodes = g1->num_nodes;
    int current_score = calculate_score_change(g1, g2, matching, -1, -1);

    srand(time(NULL));  // Initialize random seed

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int node1 = rand() % num_nodes;
        int node2 = rand() % num_nodes;
        if (node1 == node2) continue;

        int temp = matching[node1];
        matching[node1] = matching[node2];
        matching[node2] = temp;

        int delta_score = calculate_score_change(g1, g2, matching, node1, node2);
        int new_score = current_score + delta_score;
        
        if (new_score < current_score) {
            current_score = new_score;
        } else {
            matching[node2] = matching[node1];
            matching[node1] = temp;
        }
    }
}

// Write final matching to a CSV file
void write_matching(const char *filename, int *matching, int num_nodes, NodeMap *map_g1, NodeMap *map_g2) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "Node G1,Node G2\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(file, "%d,%d\n", map_g1->node_ids[i], map_g2->node_ids[matching[i]]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <initial_matching.csv> <output_matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *g1_filename = argv[1];
    const char *g2_filename = argv[2];
    const char *initial_matching_filename = argv[3];
    const char *output_filename = argv[4];

    // Create node maps for g1 and g2
    NodeMap *map_g1 = create_node_map(100);
    NodeMap *map_g2 = create_node_map(100);

    // Load the graphs
    Graph *g1 = load_graph(g1_filename, map_g1);
    if (!g1) {
        free_node_map(map_g1);
        free_node_map(map_g2);
        return EXIT_FAILURE;
    }

    Graph *g2 = load_graph(g2_filename, map_g2);
    if (!g2) {
        free_graph(g1);
        free_node_map(map_g1);
        free_node_map(map_g2);
        return EXIT_FAILURE;
    }

    // Load the initial matching
    int *matching = load_initial_matching(initial_matching_filename, map_g1, map_g2);
    if (!matching) {
        fprintf(stderr, "Error loading initial matching.\n");
        free_graph(g1);
        free_graph(g2);
        free_node_map(map_g1);
        free_node_map(map_g2);
        return EXIT_FAILURE;
    }

    // Refine the matching iteratively
    refine_matching(g1, g2, matching);

    // Write the final refined matching to the output file
    write_matching(output_filename, matching, g1->num_nodes, map_g1, map_g2);

    // Clean up
    free(matching);
    free_graph(g1);
    free_graph(g2);
    free_node_map(map_g1);
    free_node_map(map_g2);

    return EXIT_SUCCESS;
}
