#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define SAVE_INTERVAL 25
#define UPDATE_INTERVAL 20
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MAX_NODES 100000

const int NUM_NODES = 18524;

typedef enum {
    LOG_LEVEL_INFO = 0,
    LOG_LEVEL_DEBUG = 1,
    LOG_LEVEL_ERROR = 2
} LogLevel;

#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO

#define LOG_ERROR(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_ERROR) { \
        fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
    }

#define LOG_INFO(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_INFO) { \
        printf("[INFO] " fmt "\n", ##__VA_ARGS__); \
    }

#define LOG_DEBUG(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_DEBUG) { \
        printf("[DEBUG] %s: " fmt "\n", __func__, ##__VA_ARGS__); \
    }

// Structure definitions
typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap* edges;
    EdgeMap* reverse_edges;
    short** adj_matrix;
    int node_capacity;
} Graph;

typedef struct NodeMetrics {
    int in_degree;
    int out_degree;
    int total_weight;
    double avg_in_weight;
    double avg_out_weight;
    int ordering_rank;
} NodeMetrics;

// Progress bar function
void print_progress(int current, int total, const char* prefix) {
    const int bar_width = 50;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);
    
    printf("\r%s [", prefix);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
    if (current == total) printf("\n");
}

// Function to format numbers with commas
char* format_number(int num) {
    static char formatted[32];
    char temp[32];
    int i = 0, j = 0;
    
    sprintf(temp, "%d", num);
    int len = strlen(temp);
    
    while (len > 0) {
        if (i > 0 && i % 3 == 0) formatted[j++] = ',';
        formatted[j++] = temp[len - 1];
        len--;
        i++;
    }
    formatted[j] = '\0';
    
    // Reverse the string
    for (i = 0; i < j/2; i++) {
        char t = formatted[i];
        formatted[i] = formatted[j-1-i];
        formatted[j-1-i] = t;
    }
    
    return formatted;
}

EdgeMap* new_edge_map() {
    EdgeMap* em = malloc(sizeof(EdgeMap));
    if (!em) {
        LOG_ERROR("Failed to allocate EdgeMap");
        exit(1);
    }
    em->capacity = 100;
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        LOG_ERROR("Failed to allocate EdgeMap arrays");
        exit(1);
    }
    return em;
}

Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
    if (!g) {
        LOG_ERROR("Failed to allocate Graph");
        exit(1);
    }
    g->edges = NULL;
    g->reverse_edges = NULL;
    g->adj_matrix = NULL;
    g->adj_matrix = (short**)malloc((NUM_NODES+1) * sizeof(short*));
    for (int i = 0; i <= NUM_NODES; ++i) {
        g->adj_matrix[i] = (short*)calloc((NUM_NODES+1), sizeof(short)); // Initialize to 0
    }

    //g->nodes = malloc(sizeof(int) * 10000);
    //if (!g->nodes) {
    //    LOG_ERROR("Failed to allocate nodes array");
    //    exit(1);
    //}
    //g->node_count = 0;
    g->node_capacity = 10000;
    return g;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            LOG_ERROR("Failed to reallocate EdgeMap arrays");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

//void add_node(Graph* g, int node) {
//    for (int i = 0; i < g->node_count; i++) {
//        if (g->nodes[i] == node) return;
//    }
//    
//    if (g->node_count >= g->node_capacity) {
//        g->node_capacity *= 2;
//        int* new_nodes = realloc(g->nodes, sizeof(int) * g->node_capacity);
//        if (!new_nodes) {
//            LOG_ERROR("Failed to reallocate nodes array");
//            exit(1);
//        }
//        g->nodes = new_nodes;
//    }
//    g->nodes[g->node_count++] = node;
//}

void add_edge(Graph* g, int from, int to, int weight) {
  //    LOG_DEBUG("Adding edge: %d -> %d (weight: %d)", from, to, weight);
    g->adj_matrix[from][to] = weight;
    if (g->edges == NULL) {
        g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
        g->reverse_edges = calloc(MAX_NODES, sizeof(EdgeMap));
        if (!g->edges || !g->reverse_edges) {
            LOG_ERROR("Failed to allocate edges arrays");
            exit(1);
        }
    }
    
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
}

int get_weight(Graph* g, int from, int to) {
    return g->adj_matrix[from][to];
    //for (int i = 0; i < g->edges[from].count; i++) {
    //    if (g->edges[from].to_nodes[i] == to) {
    //        return g->edges[from].weights[i];
    //    }
    //}
    //return 0;
}

int* read_ordering(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Error opening file: %s", filename);
        exit(1);
    }

    int* ordering = calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    
    LOG_INFO("Reading ordering from %s", filename);
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int node_id, order;
        sscanf(line, "%d,%d", &node_id, &order);
        ordering[node_id] = order;
    }
    
    fclose(file);
    return ordering;
}

NodeMetrics* calculate_node_metrics(Graph* g, const char* ordering_path) {
    int max_node = NUM_NODES;
    //for (int i = 0; i < g->node_count; i++) {
    //    if (g->nodes[i] > max_node) max_node = g->nodes[i];
    //}

    LOG_INFO("Calculating node metrics");
    
    int* ordering = read_ordering(ordering_path, max_node);
    NodeMetrics* metrics = calloc(max_node + 1, sizeof(NodeMetrics));
    
    for (int node=1; node<=NUM_NODES; node++) {
        //int node = g->nodes[i];
        NodeMetrics* m = &metrics[node];
        
        m->out_degree = g->edges[node].count;
        for (int j = 0; j < g->edges[node].count; j++) {
            m->total_weight += g->edges[node].weights[j];
        }
        
        m->in_degree = g->reverse_edges[node].count;
        for (int j = 0; j < g->reverse_edges[node].count; j++) {
            m->total_weight += g->reverse_edges[node].weights[j];
        }
        
        if (m->out_degree > 0) {
            m->avg_out_weight = (double)m->total_weight / m->out_degree;
        }
        if (m->in_degree > 0) {
            m->avg_in_weight = (double)m->total_weight / m->in_degree;
        }
        
        m->ordering_rank = ordering[node];
    }
    
    free(ordering);
    return metrics;
}

double calculate_node_similarity(NodeMetrics m1, NodeMetrics m2) {
    double score = 5 * fabs(m1.in_degree - m2.in_degree) +
                   5 * fabs(m1.out_degree - m2.out_degree) +
                   fabs(m1.avg_in_weight - m2.avg_in_weight) +
                   fabs(m1.avg_out_weight - m2.avg_out_weight);
    
    double ordering_sim = fabs(m1.ordering_rank - m2.ordering_rank);
    score = 0.7 * score + 0.3 * ordering_sim;
    
    return -score;
}

int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int score = 0;
    
    for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
        //int src_m = gm->i;
        for (int j = 0; j < gm->edges[src_m].count; j++) {
            int dst_m = gm->edges[src_m].to_nodes[j];
            int weight_m = gm->edges[src_m].weights[j];
            int src_f = mapping[src_m];
            int dst_f = mapping[dst_m];
            score += MIN(weight_m, gf->adj_matrix[src_f][dst_f]);
        }
    }
    
    return score;
}

void validate_mapping_changes(int* old_mapping, int* new_mapping, int max_node,
                            int node_m1, int node_m2) {
    for (int i = 1; i <= max_node; i++) {
        if (i != node_m1 && i != node_m2) {
            if (old_mapping[i] != new_mapping[i]) {
                LOG_ERROR("Unexpected mapping change for node %d: %d -> %d",
                         i, old_mapping[i], new_mapping[i]);
            }
        }
    }
}

int calculate_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2) {
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int delta = 0;
    
    // Handle outgoing edges from node_m1
    for (int i = 0; i < gm->edges[node_m1].count; i++) {
        int dst_m = gm->edges[node_m1].to_nodes[i];
        if (dst_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m1].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int dst_m = gm->edges[node_m2].to_nodes[i];
        if (dst_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m2].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m2
    for (int i = 0; i < gm->reverse_edges[node_m2].count; i++) {
        int src_m = gm->reverse_edges[node_m2].to_nodes[i];
        if (src_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m2].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle direct edges between the swapped nodes
    // From m1 to m2
    int m1_to_m2 = gm->adj_matrix[node_m1][node_m2];
    if (m1_to_m2 > 0) {
        int old_weight = MIN(m1_to_m2, gf->adj_matrix[node_f1][node_f2]);
        int new_weight = MIN(m1_to_m2, gf->adj_matrix[node_f2][node_f1]);
        delta += new_weight - old_weight;
    }
    
    // From m2 to m1
    int m2_to_m1 = gm->adj_matrix[node_m2][node_m1];
    if (m2_to_m1 > 0) {
        int old_weight = MIN(m2_to_m1, gf->adj_matrix[node_f2][node_f1]);
        int new_weight = MIN(m2_to_m1, gf->adj_matrix[node_f1][node_f2]);
        delta += new_weight - old_weight;
    }
    
    return delta;
}

void write_mapping(const char* filename, int* mapping, int max_node) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        exit(1);
    }
    
    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 1; i <= max_node; i++) {
        if (mapping[i] != 0) {
            fprintf(file, "m%d,f%d\n", i, mapping[i]);
        }
    }
    
    fclose(file);
}

// Function to load benchmark mapping from CSV
int* load_benchmark_mapping(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    int* mapping = calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // Try to get expected score from filename
    int expected_score = 0;
    const char* underscore = strrchr(filename, '_');
    if (underscore) {
        expected_score = atoi(underscore + 1);
        LOG_INFO("Expected score from filename: %d", expected_score);
    }
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    LOG_INFO("Loading benchmark mapping from %s", filename);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int male_id, female_id;
        
        // Try direct integer format first
        if (sscanf(line, "%d,%d", &male_id, &female_id) == 2) {
            mapping[male_id] = female_id;
            count++;
        } else {
            // Try format with prefixes
            char male_str[20], female_str[20];
            if (sscanf(line, "%[^,],%s", male_str, female_str) == 2) {
                male_id = atoi(male_str + (male_str[0] == 'm' ? 1 : 0));
                female_id = atoi(female_str + (female_str[0] == 'f' ? 1 : 0));
                mapping[male_id] = female_id;
                count++;
            }
        }
    }
    
    LOG_INFO("Loaded %s mappings", format_number(count));
    
    fclose(file);
    return mapping;
}

// Save intermediate mapping with verification
void save_intermediate_mapping(const char* filename, int* mapping, int max_node, 
                             Graph* gm, Graph* gf, int current_score) {
    write_mapping(filename, mapping, max_node);
    
    // Verify written mapping
    int* verification = load_benchmark_mapping(filename, max_node);
    if (verification) {
        int verify_score = calculate_alignment_score(gm, gf, verification);
        if (verify_score != current_score) {
            LOG_ERROR("Score mismatch - internal: %d, written: %d", 
                     current_score, verify_score);
        }
        free(verification);
    }
}


// Function to get maximum node ID from graph
int get_max_node(Graph* g) {
    return NUM_NODES;
}

// Function to clean up graph memory
void free_graph(Graph* g) {
    if (g->edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
        }
        free(g->edges);
    }
    
    if (g->reverse_edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        free(g->reverse_edges);
    }
    if (g->adj_matrix){
        for (int i=0; i<=NUM_NODES; i++){
            free(g->adj_matrix[i]);
        }
        free(g->adj_matrix);
        g->adj_matrix=NULL;
    }
    //free(g->nodes);
    free(g);
}

Graph* load_graph_from_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    Graph* graph = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int total_lines = 0;
    
    // Count total lines for progress bar
    while (fgets(line, MAX_LINE_LENGTH, file)) total_lines++;
    rewind(file);
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    total_lines--; // Adjust for header
    
    LOG_INFO("Loading graph from %s (%s lines)", filename, format_number(total_lines));
    
    time_t start_time = time(NULL);
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(graph, from, to, weight);
            line_count++;
            if (line_count % 100000 == 0) {
                print_progress(line_count, total_lines, "Loading graph");
            }
        } else {
            LOG_ERROR("Malformed line in CSV: %s", line);
        }
    }
    
    time_t end_time = time(NULL);
    LOG_INFO("Graph loaded successfully:");
    LOG_INFO("  - Nodes: %s", format_number(NUM_NODES));
    LOG_INFO("  - Edges: %s", format_number(line_count));
    LOG_INFO("  - Time taken: %ld seconds", end_time - start_time);
    
    fclose(file);
    return graph;
}


// Helper to get the next lexicographical permutation
bool next_permutation(int* arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i + 1]) i--;
    if (i < 0) return false;
    int j = n - 1;
    while (arr[j] <= arr[i]) j--;
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    for (int left = i + 1, right = n - 1; left < right; left++, right--) {
        temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }
    return true;
}

// Helper function to swap elements in an array
void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}



// Define a structure for edges to facilitate uniqueness checking
typedef struct {
    int src;
    int dst;
    int weight;
} UniqueEdge;

// Function to check if an edge already exists in the unique edges array
bool edge_exists(UniqueEdge* edges, int edge_count, int src, int dst) {
    for (int i = 0; i < edge_count; i++) {
        if (edges[i].src == src && edges[i].dst == dst) {
            return true;
        }
    }
    return false;
}

int calculate_permutation_delta(Graph* gm, Graph* gf, int* mapping, int* vertex_set, int set_size) {
    int delta = 0;
    int initial_capacity = 100;
    int edge_count = 0;

    // Dynamically allocate memory for unique edges with an initial capacity
    UniqueEdge* unique_edges = (UniqueEdge*)malloc(initial_capacity * sizeof(UniqueEdge));
    if (!unique_edges) {
        fprintf(stderr, "Memory allocation failed for unique_edges.\n");
        exit(1);
    }

    // Collect unique edges connected to the vertices in vertex_set
    for (int i = 0; i < set_size; i++) {
        int node_m = vertex_set[i];

        // Outgoing edges from node_m
        for (int j = 0; j < gm->edges[node_m].count; j++) {
            int dst_m = gm->edges[node_m].to_nodes[j];
            int weight_m = gm->edges[node_m].weights[j];

            if (!edge_exists(unique_edges, edge_count, node_m, dst_m)) {
                // Resize array if capacity is reached
                if (edge_count == initial_capacity) {
                    initial_capacity *= 2;
                    unique_edges = (UniqueEdge*)realloc(unique_edges, initial_capacity * sizeof(UniqueEdge));
                    if (!unique_edges) {
                        fprintf(stderr, "Memory reallocation failed for unique_edges.\n");
                        exit(1);
                    }
                }
                unique_edges[edge_count++] = (UniqueEdge){node_m, dst_m, weight_m};
            }
        }

        // Incoming edges to node_m
        for (int j = 0; j < gm->reverse_edges[node_m].count; j++) {
            int src_m = gm->reverse_edges[node_m].to_nodes[j];
            int weight_m = gm->reverse_edges[node_m].weights[j];

            if (!edge_exists(unique_edges, edge_count, src_m, node_m)) {
                // Resize array if capacity is reached
                if (edge_count == initial_capacity) {
                    initial_capacity *= 2;
                    unique_edges = (UniqueEdge*)realloc(unique_edges, initial_capacity * sizeof(UniqueEdge));
                    if (!unique_edges) {
                        fprintf(stderr, "Memory reallocation failed for unique_edges.\n");
                        exit(1);
                    }
                }
                unique_edges[edge_count++] = (UniqueEdge){src_m, node_m, weight_m};
            }
        }
    }

    // Calculate the score for each unique edge based on old and new mappings
    for (int i = 0; i < edge_count; i++) {
        int src_m = unique_edges[i].src;
        int dst_m = unique_edges[i].dst;
        int weight_m = unique_edges[i].weight;

        int new_src_f = mapping[src_m];
        int new_dst_f = mapping[dst_m];

        // Calculate new contribution for each unique edge
        delta += MIN(weight_m, gf->adj_matrix[new_src_f][new_dst_f]);
    }

    // Free the allocated memory for unique edges
    free(unique_edges);

    // Return the score difference after applying the permutation
    return delta;
}



// Recursive function to generate all permutations of k nodes
void generate_permutations(Graph* gm, Graph* gf, int* current_mapping, int * originalnodes, int* nodes, int k, int depth, int* best_delta, int* improvements, const char* out_path, int max_node) {
    if (depth == k) {
        // Apply the permutation to a temporary mapping
        int* temp_mapping = malloc(sizeof(int) * (max_node + 1));
        memcpy(temp_mapping, current_mapping, sizeof(int) * (max_node + 1));

        // Apply the new permutation of nodes to temp_mapping
        for (int i = 0; i < k; i++) {
            temp_mapping[nodes[i]] = originalnodes[i+k];
        }
        // Calculate the score for this new mapping
        int new_delta = calculate_permutation_delta( gm, gf, temp_mapping, nodes, k);
        // If the new score is better, update the best mapping
        if (new_delta > *best_delta) {

            int oldscore=calculate_alignment_score(gm, gf, current_mapping);
            int myscore=calculate_alignment_score(gm, gf, temp_mapping);
	    if (myscore-oldscore >0) {
                   *best_delta = new_delta;
                   #pragma omp critical 
		   {
                       memcpy(current_mapping, temp_mapping, sizeof(int) * (max_node + 1));
                       (*improvements)++;
		   }
                   // Save the improved mapping to disk
                   LOG_DEBUG("Improved mapping found with score: %s", format_number(myscore));
                   save_intermediate_mapping(out_path, current_mapping, max_node, gm, gf, myscore);
	    } else {
                   LOG_DEBUG("Delta calculation wrong!");
	    }
        }

        free(temp_mapping);
        return;
    }

    // Generate permutations by recursively swapping elements
    for (int i = depth; i < k; i++) {
        swap(nodes, depth, i);
        generate_permutations(gm, gf, current_mapping, originalnodes, nodes, k, depth + 1, best_delta, improvements, out_path, max_node);

        swap(nodes, depth, i); // backtrack
    }
}



void process_permutation(Graph* gm, Graph* gf, int* current_mapping, int* originalnodes, int* nodes, int k, int* best_delta, int* improvements, const char* out_path, int max_node) {
	
    // Use generate_permutations to test all permutations of k nodes
    generate_permutations(gm, gf, current_mapping, originalnodes, nodes, k, 0, best_delta, improvements, out_path, max_node);
}



// Helper to generate combinations iteratively
void generate_combinations_iterative(int n, int k, Graph* gm, Graph* gf, int* current_mapping, int* current_delta, int* improvements, const char* out_path, int max_node) {
    int* combo = malloc(sizeof(int) * k);
    int* originalcombo = malloc(sizeof(int) * k*2);

    time_t start_time = time(NULL);
    int num_combination=0;
    for (int i = 0; i < k; i++) {
        combo[i] = i + 1; // Initialize the first combination
	originalcombo[i]=i+1;
	originalcombo[k+i]=current_mapping[i+1];
    }
    *current_delta = calculate_permutation_delta( gm, gf, current_mapping, combo, k);


    while (true) {
        // Process each permutation of the current combination
        process_permutation(gm, gf, current_mapping, originalcombo,combo, k, current_delta, improvements, out_path, max_node);

        // Generate the next combination
        int i = k - 1;
        while (i >= 0 && combo[i] == n - k + i + 1) i--;
        if (i < 0) break;
        combo[i]++;
        for (int j = i + 1; j < k; j++) {
            combo[j] = combo[j - 1] + 1;
        }

        //printf("Generate new combination\n");
        for (int i = 0; i < k; i++) {
	    originalcombo[i]=combo[i];
	    originalcombo[k+i]=current_mapping[combo[i]];
	    //printf("%d,",combo[i]);
	}
	/*
        printf("new mapping is \n");
        for (int i = 0; i < k; i++) {
	    printf("%d,",originalcombo[i+k]);
	}
	*/




	int old_delta=*current_delta;
	*current_delta= calculate_permutation_delta( gm, gf, current_mapping, combo, k);
        //printf("\n new delta is %d\n",*current_delta);

	num_combination++;
	if (num_combination % 1000 ==0 ) {
	    time_t current_time = time(NULL);
            double elapsed = difftime(current_time, start_time);
            double coms_per_sec = num_combination / elapsed;

            int current_score=calculate_alignment_score(gm, gf, current_mapping);
            LOG_INFO("Optimization progress:");
            LOG_INFO("  - Processed: %s combinations ",
                    format_number(num_combination));
            LOG_INFO("  - Current score: %s", format_number(current_score));
            LOG_INFO("  - Improvements found: %s", format_number(*improvements));
            LOG_INFO("  - Processing speed: %.1f combinations/sec", coms_per_sec);

	}
    }
    free(combo);
}




// Structure to hold combination data
typedef struct {
    int* combo;
    int* originalcombo;
    int delta;
} CombinationData;

// Function to generate next combination
bool get_next_combination(int* combo, int n, int k) {
    int i = k - 1;
    while (i >= 0 && combo[i] == n - k + i + 1) i--;
    if (i < 0) return false;
    
    combo[i]++;
    for (int j = i + 1; j < k; j++) {
        combo[j] = combo[j - 1] + 1;
    }
    return true;
}

// Function to generate and process combinations in parallel
void generate_combinations_parallel(int n, int k, Graph* gm, Graph* gf, int* current_mapping, 
                                 int* current_delta, int* improvements, const char* out_path, int max_node) {
    const int BATCH_SIZE = 256;
    CombinationData* combinations = malloc(sizeof(CombinationData) * BATCH_SIZE);
    time_t start_time = time(NULL);
    int num_combination = 0;
    
    // Initialize combinations array
    for (int i = 0; i < BATCH_SIZE; i++) {
        combinations[i].combo = malloc(sizeof(int) * k);
        combinations[i].originalcombo = malloc(sizeof(int) * k * 2);
    }
    
    // Initialize first combination
    for (int i = 0; i < k; i++) {
        combinations[0].combo[i] = i + 1;
        combinations[0].originalcombo[i] = i + 1;
        combinations[0].originalcombo[k+i] = current_mapping[i+1];
    }
    combinations[0].delta = calculate_permutation_delta(gm, gf, current_mapping, combinations[0].combo, k);
    
    while (true) {
        int batch_count = 1;  // First combination is already set
        
        // Generate batch of combinations
        while (batch_count < BATCH_SIZE) {
            for (int i = 0; i < k; i++) {
                combinations[batch_count].combo[i] = combinations[batch_count-1].combo[i];
            }
            if (!get_next_combination(combinations[batch_count].combo, n, k)) break;
            
            // Copy the new combination
            for (int i = 0; i < k; i++) {
                combinations[batch_count].originalcombo[i] = combinations[batch_count].combo[i];
                combinations[batch_count].originalcombo[k+i] = current_mapping[combinations[batch_count].combo[i]];
            }
            
            // Calculate delta for new combination
            combinations[batch_count].delta = calculate_permutation_delta(gm, gf, current_mapping, combinations[batch_count].combo, k);
            batch_count++;
        }
        
        if (batch_count == 0) break;  // No more combinations to process
        
        // Process combinations in parallel
        #pragma omp parallel for  schedule(dynamic)
        for (int i = 0; i < batch_count; i++) {
            process_permutation(gm, gf, current_mapping, combinations[i].originalcombo,
                              combinations[i].combo, k, &combinations[i].delta, 
                              improvements, out_path, max_node);
        }
        
        num_combination += batch_count;
        
        // Log progress every 1000 combinations
        if ((num_combination) % 1024 == 0) {
            time_t current_time = time(NULL);
            double elapsed = difftime(current_time, start_time);
            double coms_per_sec = num_combination / elapsed;
            int current_score = calculate_alignment_score(gm, gf, current_mapping);
            
            LOG_INFO("Optimization progress:");
            LOG_INFO("  - Processed: %s combinations ", format_number(num_combination));
            LOG_INFO("  - Current score: %s", format_number(current_score));
            LOG_INFO("  - Improvements found: %s", format_number(*improvements));
            LOG_INFO("  - Processing speed: %.1f combinations/sec", coms_per_sec);
        }
        
        // If we couldn't fill the batch, we're done
        if (batch_count < BATCH_SIZE) break;
        if (!get_next_combination(combinations[batch_count-1].combo, n, k)) break;
        for (int i = 0; i < k; i++) {
                combinations[0].combo[i] = combinations[batch_count-1].combo[i];
        }
        for (int i = 0; i < k; i++) {
                combinations[0].originalcombo[i] = combinations[0].combo[i];
                combinations[0].originalcombo[k+i] = current_mapping[combinations[0].combo[i]];
        }
            
        // Calculate delta for new combination
        combinations[0].delta = calculate_permutation_delta(gm, gf, current_mapping, combinations[0].combo, k);
    }
    
    // Cleanup
    for (int i = 0; i < BATCH_SIZE; i++) {
        free(combinations[i].combo);
        free(combinations[i].originalcombo);
    }
    free(combinations);
}




// Optimizes mapping with intermediate saving
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping, const char* male_ordering_path, const char* female_ordering_path, const char* out_path) {
    int max_node = NUM_NODES;
    int k = 4; // Adjust the value of k as needed

    int* current_mapping = malloc(sizeof(int) * (max_node + 1));
    memcpy(current_mapping, initial_mapping, sizeof(int) * (max_node + 1));

    NodeMetrics* metrics_m = calculate_node_metrics(gm, male_ordering_path);
    NodeMetrics* metrics_f = calculate_node_metrics(gf, female_ordering_path);

    int current_delta=0;
    int improvements = 0;
    time_t start_time = time(NULL);
    int current_score = calculate_alignment_score(gm, gf, current_mapping);

    LOG_INFO("Starting optimization with initial score: %s", format_number(current_score));

    // Iteratively generate combinations and process each permutation
    //generate_combinations_iterative(NUM_NODES, k, gm, gf, current_mapping, &current_delta, &improvements, out_path, max_node);
    while (true) {
        generate_combinations_parallel (NUM_NODES, k, gm, gf, current_mapping, &current_delta, &improvements, out_path, max_node);
    }
    current_score = calculate_alignment_score(gm, gf, current_mapping);
    // Final log information
    time_t end_time = time(NULL);
    LOG_INFO("Optimization completed:");
    LOG_INFO("  - Final score: %s", format_number(current_score));
    LOG_INFO("  - Total improvements: %s", format_number(improvements));
    LOG_INFO("  - Time taken: %.1f minutes", difftime(end_time, start_time) / 60);

    save_intermediate_mapping(out_path, current_mapping, max_node, gm, gf, current_score);

    free(metrics_m);
    free(metrics_f);
    return current_mapping;
}



// Main function
int main(int argc, char* argv[]) {
    if (argc < 7) {
        LOG_ERROR("Usage: %s <male graph> <female graph> <male ordering> <female ordering> <in mapping> <out mapping>", argv[0]);
        return 1;
    }
    
    time_t total_start = time(NULL);
    LOG_INFO("Graph Alignment Tool v1.0");
    LOG_INFO("Starting process with:");
    LOG_INFO("  - Male graph: %s", argv[1]);
    LOG_INFO("  - Female graph: %s", argv[2]);
    LOG_INFO("  - Output mapping: %s", argv[6]);
    
    Graph* gm = load_graph_from_csv(argv[1]);
    if (!gm) {
        LOG_ERROR("Failed to load male graph");
        return 1;
    }
    
    Graph* gf = load_graph_from_csv(argv[2]);
    if (!gf) {
        LOG_ERROR("Failed to load female graph");
        free_graph(gm);
        return 1;
    }
    
    int max_node = MAX(get_max_node(gm), get_max_node(gf));
    
    int* benchmark = load_benchmark_mapping(argv[5], max_node);
    if (!benchmark) {
        LOG_ERROR("Failed to load benchmark mapping");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }
    
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    LOG_INFO("Initial alignment score: %s", format_number(initial_score));
    
    int* optimized_mapping = optimize_mapping(gm, gf, benchmark, argv[3], argv[4], argv[6]);
    int optimized_score = calculate_alignment_score(gm, gf, optimized_mapping);
    
    time_t total_end = time(NULL);
    LOG_INFO("Process completed:");
    LOG_INFO("  - Initial score: %s", format_number(initial_score));
    LOG_INFO("  - Final score: %s", format_number(optimized_score));
    LOG_INFO("  - Improvement: %.2f%%",
            (double)(optimized_score - initial_score) / initial_score * 100.0);
    LOG_INFO("  - Total time: %.1f minutes", difftime(total_end, total_start) / 60);
    
    free_graph(gm);
    free_graph(gf);
    free(benchmark);
    free(optimized_mapping);
    
    return 0;
}
