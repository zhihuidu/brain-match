#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int source;
    int target;
    int weight;
} Edge;

typedef struct {
    int num_edges;
    Edge *edges;
} Graph;

typedef struct {
    int *node_ids;
    int count;
    int capacity;
} NodeMap;

int binary_search(int *array, int size, int value) {
    int low = 0, high = size - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (array[mid] == value) return mid;
        else if (array[mid] < value) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

int add_node_id(NodeMap *map, int node_id) {
    int idx = binary_search(map->node_ids, map->count, node_id);
    if (idx != -1) return idx;

    if (map->count >= map->capacity) {
        map->capacity *= 2;
        map->node_ids = realloc(map->node_ids, map->capacity * sizeof(int));
    }

    int pos = map->count;
    while (pos > 0 && map->node_ids[pos - 1] > node_id) {
        map->node_ids[pos] = map->node_ids[pos - 1];
        pos--;
    }
    map->node_ids[pos] = node_id;
    map->count++;
    return pos;
}

NodeMap *create_node_map(int initial_capacity) {
    NodeMap *map = malloc(sizeof(NodeMap));
    map->node_ids = malloc(initial_capacity * sizeof(int));
    map->count = 0;
    map->capacity = initial_capacity;
    return map;
}

void free_node_map(NodeMap *map) {
    free(map->node_ids);
    free(map);
}

int count_graph_edges(const char *filename, NodeMap *map) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return -1;
    }

    int num_edges = 0;
    char line[256];
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) == 3) {
            add_node_id(map, src);
            add_node_id(map, tgt);
            num_edges++;
        }
    }

    fclose(file);
    return num_edges;
}

Graph *load_graph(const char *filename, int num_edges, NodeMap *map) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    Graph *graph = malloc(sizeof(Graph));
    graph->num_edges = num_edges;
    graph->edges = malloc(num_edges * sizeof(Edge));

    char line[256];
    fgets(line, sizeof(line), file);

    int i = 0;
    while (fgets(line, sizeof(line), file) && i < num_edges) {
        int src, tgt, weight;
        sscanf(line, "%d,%d,%d", &src, &tgt, &weight);

        graph->edges[i].source = binary_search(map->node_ids, map->count, src);
        graph->edges[i].target = binary_search(map->node_ids, map->count, tgt);
        graph->edges[i].weight = weight;
        i++;
    }

    fclose(file);
    return graph;
}

int *load_matching(const char *filename, NodeMap *map) {
    int *matching = malloc(map->count * sizeof(int));
    if (!matching) {
        fprintf(stderr, "Memory allocation error for matching array.\n");
        return NULL;
    }

    for (int i = 0; i < map->count; i++) {
        matching[i] = -1;
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening matching file: %s\n", filename);
        free(matching);
        return NULL;
    }

    char line[256];
    fgets(line, sizeof(line), file);

    int node_g1, node_g2;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d,%d", &node_g1, &node_g2) == 2) {
            int idx_g1 = binary_search(map->node_ids, map->count, node_g1);
            int idx_g2 = binary_search(map->node_ids, map->count, node_g2);
            if (idx_g1 != -1 && idx_g2 != -1) {
                matching[idx_g1] = idx_g2;
            }
        }
    }

    fclose(file);
    return matching;
}

int find_and_remove_edge(Graph *graph, int source, int target) {
    for (int i = 0; i < graph->num_edges; i++) {
        if (graph->edges[i].source == source && graph->edges[i].target == target) {
            int weight = graph->edges[i].weight;
            graph->edges[i] = graph->edges[--graph->num_edges];
            return weight;
        }
    }
    return -1;
}

int calculate_score(Graph *g1, Graph *g2, int *matching) {
    int score = 0;

    for (int i = 0; i < g1->num_edges; i++) {
        int source_g1 = g1->edges[i].source;
        int target_g1 = g1->edges[i].target;
        int weight_g1 = g1->edges[i].weight;

        int source_g2 = matching[source_g1];
        int target_g2 = matching[target_g1];

        if (source_g2 != -1 && target_g2 != -1) {
            int weight_g2 = find_and_remove_edge(g2, source_g2, target_g2);
            if (weight_g2 != -1) {
                score += abs(weight_g1 - weight_g2);
            } else {
                score += weight_g1;
            }
        } else {
            score += weight_g1;
        }
    }

    for (int i = 0; i < g2->num_edges; i++) {
        score += g2->edges[i].weight;
    }

    return score;
}

void free_graph(Graph *graph) {
    free(graph->edges);
    free(graph);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *g1_filename = argv[1];
    const char *g2_filename = argv[2];
    const char *matching_filename = argv[3];

    NodeMap *map = create_node_map(1000);

    int g1_num_edges = count_graph_edges(g1_filename, map);
    if (g1_num_edges < 0) {
        free_node_map(map);
        return EXIT_FAILURE;
    }

    Graph *g1 = load_graph(g1_filename, g1_num_edges, map);
    if (!g1) {
        free_node_map(map);
        return EXIT_FAILURE;
    }

    int g2_num_edges = count_graph_edges(g2_filename, map);
    if (g2_num_edges < 0) {
        free_graph(g1);
        free_node_map(map);
        return EXIT_FAILURE;
    }

    Graph *g2 = load_graph(g2_filename, g2_num_edges, map);
    if (!g2) {
        free_graph(g1);
        free_node_map(map);
        return EXIT_FAILURE;
    }

    int *matching = load_matching(matching_filename, map);
    if (!matching) {
        fprintf(stderr, "Error loading matching file.\n");
        free_graph(g1);
        free_graph(g2);
        free_node_map(map);
        return EXIT_FAILURE;
    }

    int score = calculate_score(g1, g2, matching);
    printf("Match score: %d\n", score);

    free_graph(g1);
    free_graph(g2);
    free(matching);
    free_node_map(map);

    return EXIT_SUCCESS;
}
