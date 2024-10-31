#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

// Error codes
#define SUCCESS 0
#define ERROR_FILE_OPEN -1
#define ERROR_MEMORY_ALLOC -2
#define ERROR_INVALID_INPUT -3

typedef struct {
    int source;
    int target;
    int weight;
} Edge;

typedef struct {
    size_t num_edges;
    size_t capacity;
    Edge *edges;
} Graph;

typedef struct {
    int *node_ids;
    size_t count;
    size_t capacity;
} NodeMap;

// Function prototypes with error handling
int binary_search(const int *array, size_t size, int value);
NodeMap *create_node_map(size_t initial_capacity);
void free_node_map(NodeMap *map);
int add_node_id(NodeMap *map, int node_id);
Graph *create_graph(size_t initial_capacity);
void free_graph(Graph *graph);
int count_graph_edges(const char *filename, NodeMap *map, size_t *num_edges);
int load_graph(const char *filename, Graph *graph, NodeMap *map);
int *create_matching(const NodeMap *map);
int load_matching(const char *filename, NodeMap *map, int *matching);
int calculate_score(const Graph *g1, Graph *g2, const int *matching);

// Improved binary search implementation
int binary_search(const int *array, size_t size, int value) {
    if (!array || size == 0) return -1;
    
    size_t low = 0;
    size_t high = size - 1;
    
    while (low <= high) {
        size_t mid = low + (high - low) / 2;  // Prevents potential overflow
        if (array[mid] == value) return (int)mid;
        if (array[mid] < value) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

// Improved NodeMap creation with error checking
NodeMap *create_node_map(size_t initial_capacity) {
    NodeMap *map = malloc(sizeof(NodeMap));
    if (!map) return NULL;
    
    map->node_ids = malloc(initial_capacity * sizeof(int));
    if (!map->node_ids) {
        free(map);
        return NULL;
    }
    
    map->count = 0;
    map->capacity = initial_capacity;
    return map;
}

void free_node_map(NodeMap *map) {
    if (map) {
        free(map->node_ids);
        free(map);
    }
}

// Improved node addition with bounds checking
int add_node_id(NodeMap *map, int node_id) {
    if (!map) return ERROR_INVALID_INPUT;
    
    int idx = binary_search(map->node_ids, map->count, node_id);
    if (idx != -1) return idx;
    
    if (map->count >= map->capacity) {
        size_t new_capacity = map->capacity * 2;
        int *new_ids = realloc(map->node_ids, new_capacity * sizeof(int));
        if (!new_ids) return ERROR_MEMORY_ALLOC;
        
        map->node_ids = new_ids;
        map->capacity = new_capacity;
    }
    
    size_t pos = map->count;
    while (pos > 0 && map->node_ids[pos - 1] > node_id) {
        map->node_ids[pos] = map->node_ids[pos - 1];
        pos--;
    }
    
    map->node_ids[pos] = node_id;
    map->count++;
    return (int)pos;
}

// New Graph creation function
Graph *create_graph(size_t initial_capacity) {
    Graph *graph = malloc(sizeof(Graph));
    if (!graph) return NULL;
    
    graph->edges = malloc(initial_capacity * sizeof(Edge));
    if (!graph->edges) {
        free(graph);
        return NULL;
    }
    
    graph->num_edges = 0;
    graph->capacity = initial_capacity;
    return graph;
}

void free_graph(Graph *graph) {
    if (graph) {
        free(graph->edges);
        free(graph);
    }
}

// Improved graph edge counting with better error handling
int count_graph_edges(const char *filename, NodeMap *map, size_t *num_edges) {
    FILE *file = fopen(filename, "r");
    if (!file) return ERROR_FILE_OPEN;
    
    *num_edges = 0;
    char line[256];
    int src, tgt, weight;
    
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return ERROR_INVALID_INPUT;
    }
    
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) == 3) {
            if (add_node_id(map, src) < 0 || add_node_id(map, tgt) < 0) {
                fclose(file);
                return ERROR_MEMORY_ALLOC;
            }
            (*num_edges)++;
        }
    }
    
    fclose(file);
    return SUCCESS;
}

// Improved graph loading with better error handling
int load_graph(const char *filename, Graph *graph, NodeMap *map) {
    FILE *file = fopen(filename, "r");
    if (!file) return ERROR_FILE_OPEN;
    
    char line[256];
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return ERROR_INVALID_INPUT;
    }
    
    while (fgets(line, sizeof(line), file) && graph->num_edges < graph->capacity) {
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) == 3) {
            int src_idx = binary_search(map->node_ids, map->count, src);
            int tgt_idx = binary_search(map->node_ids, map->count, tgt);
            
            if (src_idx != -1 && tgt_idx != -1) {
                graph->edges[graph->num_edges].source = src_idx;
                graph->edges[graph->num_edges].target = tgt_idx;
                graph->edges[graph->num_edges].weight = weight;
                graph->num_edges++;
            }
        }
    }
    
    fclose(file);
    return SUCCESS;
}

// New matching creation function
int *create_matching(const NodeMap *map) {
    if (!map) return NULL;
    
    int *matching = malloc(map->count * sizeof(int));
    if (matching) {
        for (size_t i = 0; i < map->count; i++) {
            matching[i] = -1;
        }
    }
    return matching;
}

// Improved matching loading with better error handling
int load_matching(const char *filename, NodeMap *map, int *matching) {
    FILE *file = fopen(filename, "r");
    if (!file) return ERROR_FILE_OPEN;
    
    char line[256];
    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return ERROR_INVALID_INPUT;
    }
    
    while (fgets(line, sizeof(line), file)) {
        int node_g1, node_g2;
        if (sscanf(line, "%d,%d", &node_g1, &node_g2) == 2) {
            int idx_g1 = binary_search(map->node_ids, map->count, node_g1);
            int idx_g2 = binary_search(map->node_ids, map->count, node_g2);
            if (idx_g1 != -1 && idx_g2 != -1) {
                matching[idx_g1] = idx_g2;
            }
        }
    }
    
    fclose(file);
    return SUCCESS;
}

// Improved score calculation
int calculate_score(const Graph *g1, Graph *g2, const int *matching) {
    if (!g1 || !g2 || !matching) return -1;
    
    int score = 0;
    
    // Process edges from g1
    for (size_t i = 0; i < g1->num_edges; i++) {
        int source_g1 = g1->edges[i].source;
        int target_g1 = g1->edges[i].target;
        int weight_g1 = g1->edges[i].weight;
        
        int source_g2 = matching[source_g1];
        int target_g2 = matching[target_g1];
        
        if (source_g2 != -1 && target_g2 != -1) {
            // Find matching edge in g2
            int found = 0;
            for (size_t j = 0; j < g2->num_edges; j++) {
                if (g2->edges[j].source == source_g2 && g2->edges[j].target == target_g2) {
                    score += abs(weight_g1 - g2->edges[j].weight);
                    // Remove the matched edge from g2
                    g2->edges[j] = g2->edges[--g2->num_edges];
                    found = 1;
                    break;
                }
            }
            if (!found) score += weight_g1;
        } else {
            score += weight_g1;
        }
    }
    
    // Add remaining edges from g2
    for (size_t i = 0; i < g2->num_edges; i++) {
        score += g2->edges[i].weight;
    }
    
    return score;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <g1.csv> <g2.csv> <matching.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    const char *g1_filename = argv[1];
    const char *g2_filename = argv[2];
    const char *matching_filename = argv[3];
    
    // Initialize structures with error handling
    NodeMap *map = create_node_map(1000);
    if (!map) {
        fprintf(stderr, "Failed to create node map\n");
        return EXIT_FAILURE;
    }
    
    size_t g1_edges, g2_edges;
    if (count_graph_edges(g1_filename, map, &g1_edges) != SUCCESS) {
        fprintf(stderr, "Error counting edges in %s\n", g1_filename);
        free_node_map(map);
        return EXIT_FAILURE;
    }
    
    Graph *g1 = create_graph(g1_edges);
    if (!g1 || load_graph(g1_filename, g1, map) != SUCCESS) {
        fprintf(stderr, "Error loading graph from %s\n", g1_filename);
        free_node_map(map);
        free_graph(g1);
        return EXIT_FAILURE;
    }
    
    if (count_graph_edges(g2_filename, map, &g2_edges) != SUCCESS) {
        fprintf(stderr, "Error counting edges in %s\n", g2_filename);
        free_node_map(map);
        free_graph(g1);
        return EXIT_FAILURE;
    }
    
    Graph *g2 = create_graph(g2_edges);
    if (!g2 || load_graph(g2_filename, g2, map) != SUCCESS) {
        fprintf(stderr, "Error loading graph from %s\n", g2_filename);
        free_node_map(map);
        free_graph(g1);
        free_graph(g2);
        return EXIT_FAILURE;
    }
    
    int *matching = create_matching(map);
    if (!matching || load_matching(matching_filename, map, matching) != SUCCESS) {
        fprintf(stderr, "Error loading matching from %s\n", matching_filename);
        free_node_map(map);
        free_graph(g1);
        free_graph(g2);
        free(matching);
        return EXIT_FAILURE;
    }
    
    int score = calculate_score(g1, g2, matching);
    if (score >= 0) {
        printf("Match score: %d\n", score);
    } else {
        fprintf(stderr, "Error calculating score\n");
    }
    
    // Cleanup
    free_graph(g1);
    free_graph(g2);
    free(matching);
    free_node_map(map);
    
    return EXIT_SUCCESS;
}
