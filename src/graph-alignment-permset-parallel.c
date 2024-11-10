#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1024
#define NUM_NODES 18524
#define GROUP_SIZE 12
#define NUM_SAMPLES 1000000
#define MAX_NODES 100000
#define SAVE_INTERVAL 25
#define MIN(a,b) ((a) < (b) ? (a) : (b))

// Structure definitions (same as before)
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

typedef struct {
    time_t start_time;
    int total_iterations;
    int total_improvements;
    int current_score;
    int initial_score;
    int best_score;
    int permutations_tried;
    int iterations_since_improvement;
    double best_delta;
    double avg_delta;
    double running_avg_delta[100];
    int delta_index;
    time_t last_improvement_time;
    struct {
        int last_hour;
        int last_10min;
        int last_1min;
    } improvements;
    double iterations_per_sec;
} ProgressStats;

typedef struct {
    int delta;
    int* permutation;
} PermutationResult;

typedef struct {
    int* vertices;
    int delta;
    int* best_permutation;
} VertexGroup;

// Forward declarations of all functions
void update_progress_stats(ProgressStats* stats, int delta, bool improvement);
ProgressStats* init_progress_stats(int initial_score);
void print_detailed_progress(ProgressStats* stats);
int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping);
int calculate_permutation_delta(Graph* gm, Graph* gf, int* current_mapping,
                              int* vertices, int* permutation, int group_size);
void evaluate_permutation_batch(Graph* gm, Graph* gf, int* current_mapping,
                              int* vertices, int** permutations, int batch_size,
                              PermutationResult* results);
void evaluate_vertex_groups(Graph* gm, Graph* gf, int* current_mapping,
                          VertexGroup* groups, int num_groups,
                          int** permutations, int total_perms);
void apply_best_permutation(Graph* gm, Graph* gf, int* current_mapping,
                          int* vertices, int* best_perm, int group_size);
int next_permutation(int* arr, int n);
int** generate_all_permutations(int n, int* total_perms);
void save_mapping(const char* filename, int* mapping, int score);
Graph* load_graph_from_csv(const char* filename);
int* load_mapping(const char* filename);


// Graph creation functions
EdgeMap* new_edge_map() {
    EdgeMap* em = malloc(sizeof(EdgeMap));
    em->capacity = 100;
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        fprintf(stderr, "Failed to allocate EdgeMap arrays\n");
        exit(1);
    }
    return em;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            fprintf(stderr, "Failed to reallocate EdgeMap arrays\n");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
    if (!g) {
        fprintf(stderr, "Failed to allocate Graph\n");
        exit(1);
    }
    g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
    g->reverse_edges = calloc(MAX_NODES, sizeof(EdgeMap));
    g->adj_matrix = malloc((NUM_NODES+1) * sizeof(short*));
    for (int i = 0; i <= NUM_NODES; i++) {
        g->adj_matrix[i] = calloc(NUM_NODES+1, sizeof(short));
    }
    if (!g->edges || !g->reverse_edges || !g->adj_matrix) {
        fprintf(stderr, "Failed to allocate graph structures\n");
        exit(1);
    }
    g->node_capacity = MAX_NODES;
    return g;
}

void add_edge(Graph* g, int from, int to, int weight) {
    g->adj_matrix[from][to] = weight;
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
}

void free_graph(Graph* g) {
    if (g) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        for (int i = 0; i <= NUM_NODES; i++) {
            free(g->adj_matrix[i]);
        }
        free(g->adj_matrix);
        free(g->edges);
        free(g->reverse_edges);
        free(g);
    }
}

// Progress tracking functions
ProgressStats* init_progress_stats(int initial_score) {
    ProgressStats* stats = malloc(sizeof(ProgressStats));
    stats->start_time = time(NULL);
    stats->total_iterations = 0;
    stats->total_improvements = 0;
    stats->current_score = initial_score;
    stats->initial_score = initial_score;
    stats->best_score = initial_score;
    stats->permutations_tried = 0;
    stats->iterations_since_improvement = 0;
    stats->best_delta = 0;
    stats->avg_delta = 0;
    stats->delta_index = 0;
    memset(stats->running_avg_delta, 0, sizeof(stats->running_avg_delta));
    stats->last_improvement_time = time(NULL);
    stats->improvements.last_hour = 0;
    stats->improvements.last_10min = 0;
    stats->improvements.last_1min = 0;
    stats->iterations_per_sec = 0;
    return stats;
}

void update_progress_stats(ProgressStats* stats, int delta, bool improvement) {
    time_t current_time = time(NULL);
    double elapsed = difftime(current_time, stats->start_time);
    
    #pragma omp critical
    {
        stats->total_iterations++;
        stats->iterations_per_sec = stats->total_iterations / (elapsed > 0 ? elapsed : 1);
        
        // Update running average delta
        stats->running_avg_delta[stats->delta_index] = delta;
        stats->delta_index = (stats->delta_index + 1) % 100;
        
        double sum_delta = 0;
        for(int i = 0; i < 100; i++) {
            sum_delta += stats->running_avg_delta[i];
        }
        stats->avg_delta = sum_delta / 100;
        
        if(improvement) {
            stats->total_improvements++;
            stats->current_score += delta;
            if(stats->current_score > stats->best_score) {
                stats->best_score = stats->current_score;
            }
            if(delta > stats->best_delta) {
                stats->best_delta = delta;
            }
            stats->iterations_since_improvement = 0;
            stats->last_improvement_time = current_time;
            
            // Update time-window improvements
            stats->improvements.last_hour++;
            stats->improvements.last_10min++;
            stats->improvements.last_1min++;
        } else {
            stats->iterations_since_improvement++;
        }
        
        // Reset time-window counters when window passes
        if(difftime(current_time, stats->last_improvement_time) > 3600) {
            stats->improvements.last_hour = 0;
        }
        if(difftime(current_time, stats->last_improvement_time) > 600) {
            stats->improvements.last_10min = 0;
        }
        if(difftime(current_time, stats->last_improvement_time) > 60) {
            stats->improvements.last_1min = 0;
        }
    }
}

void print_detailed_progress(ProgressStats* stats) {
    time_t current_time = time(NULL);
    double elapsed = difftime(current_time, stats->start_time);
    double since_last_impr = difftime(current_time, stats->last_improvement_time);
    
    printf("\n========== Progress Report ==========\n");
    printf("Time elapsed: %.1f minutes\n", elapsed / 60.0);
    printf("Time since last improvement: %.1f minutes\n", since_last_impr / 60.0);
    
    printf("\nScore Statistics:\n");
    printf("  Current score: %d\n", stats->current_score);
    printf("  Best score: %d\n", stats->best_score);
    printf("  Improvement from initial: %.2f%%\n", 
           100.0 * (stats->current_score - stats->initial_score) / stats->initial_score);
    
    printf("\nProgress Statistics:\n");
    printf("  Total iterations: %d\n", stats->total_iterations);
    printf("  Iterations per second: %.2f\n", stats->iterations_per_sec);
    printf("  Iterations since last improvement: %d\n", stats->iterations_since_improvement);
    
    printf("\nImprovement Statistics:\n");
    printf("  Total improvements: %d\n", stats->total_improvements);
    printf("  Improvements in last hour: %d\n", stats->improvements.last_hour);
    printf("  Improvements in last 10 minutes: %d\n", stats->improvements.last_10min);
    printf("  Improvements in last minute: %d\n", stats->improvements.last_1min);
    
    printf("\nDelta Statistics:\n");
    printf("  Best delta found: %.2f\n", stats->best_delta);
    printf("  Running average delta: %.2f\n", stats->avg_delta);
    
    printf("\nEstimated Rates:\n");
    printf("  Improvements per hour: %.2f\n", 
           stats->total_improvements / (elapsed / 3600.0));
    printf("  Average score increase per hour: %.2f\n",
           (stats->current_score - stats->initial_score) / (elapsed / 3600.0));
    
    printf("===================================\n\n");
    fflush(stdout);
}

// Permutation and scoring functions
int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int score = 0;
    #pragma omp parallel for reduction(+:score)
    for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
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

int calculate_permutation_delta(Graph* gm, Graph* gf, int* current_mapping,
                              int* vertices, int* permutation, int group_size) {
    int delta = 0;
    int* new_mapping = malloc((NUM_NODES + 1) * sizeof(int));
    memcpy(new_mapping, current_mapping, (NUM_NODES + 1) * sizeof(int));
    
    // Apply permutation
    for (int i = 0; i < group_size; i++) {
        new_mapping[vertices[i]] = current_mapping[vertices[permutation[i]]];
    }
    
    // Calculate delta only for affected edges
    #pragma omp parallel for reduction(+:delta)
    for (int i = 0; i < group_size; i++) {
        int v = vertices[i];
        
        // Outgoing edges
        for (int j = 0; j < gm->edges[v].count; j++) {
            int w = gm->edges[v].to_nodes[j];
            int weight = gm->edges[v].weights[j];
            delta -= MIN(weight, gf->adj_matrix[current_mapping[v]][current_mapping[w]]);
            delta += MIN(weight, gf->adj_matrix[new_mapping[v]][new_mapping[w]]);
        }
        
        // Incoming edges
        for (int j = 0; j < gm->reverse_edges[v].count; j++) {
            int u = gm->reverse_edges[v].to_nodes[j];
            int weight = gm->reverse_edges[v].weights[j];
            delta -= MIN(weight, gf->adj_matrix[current_mapping[u]][current_mapping[v]]);
            delta += MIN(weight, gf->adj_matrix[new_mapping[u]][new_mapping[v]]);
        }
    }
    
    free(new_mapping);
    return delta;
}

// Permutation generation and handling
int next_permutation(int* arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i + 1]) i--;
    if (i < 0) return 0;
    
    int j = n - 1;
    while (arr[j] <= arr[i]) j--;
    
    // Swap i and j
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    
    // Reverse suffix
    for (int k = i + 1, l = n - 1; k < l; k++, l--) {
        temp = arr[k];
        arr[k] = arr[l];
        arr[l] = temp;
    }
    
    return 1;
}

int** generate_all_permutations(int n, int* total_perms) {
    *total_perms = 1;
    for(int i = 2; i <= n; i++) {
        *total_perms *= i;
    }
    
    int** permutations = malloc(*total_perms * sizeof(int*));
    for(int i = 0; i < *total_perms; i++) {
        permutations[i] = malloc(n * sizeof(int));
    }
    
    // Initialize first permutation
    for(int i = 0; i < n; i++) {
        permutations[0][i] = i;
    }
    
    // Generate all other permutations
    for(int i = 1; i < *total_perms; i++) {
        memcpy(permutations[i], permutations[i-1], n * sizeof(int));
        next_permutation(permutations[i], n);
    }
    
    return permutations;
}

// File I/O functions
void save_mapping(const char* filename, int* mapping, int score) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error creating file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 1; i <= NUM_NODES; i++) {
        if (mapping[i] != 0) {
            fprintf(file, "m%d,f%d\n", i, mapping[i]);
        }
    }
    
    fclose(file);
    printf("Saved mapping with score %d to %s\n", score, filename);
}

Graph* load_graph_from_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }
    
    Graph* g = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    printf("Loading graph from %s...\n", filename);
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(g, from, to, weight);
            line_count++;
            if (line_count % 100000 == 0) {
                printf("\rLoaded %d edges", line_count);
                fflush(stdout);
            }
        }
    }
    printf("\nLoaded total of %d edges\n", line_count);
    
    fclose(file);
    return g;
}

int* load_mapping(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }
    
    int* mapping = calloc(NUM_NODES + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char male_str[20], female_str[20];
        if (sscanf(line, "%[^,],%s", male_str, female_str) == 2) {
            int male_id = atoi(male_str + 1);  // Skip 'm' prefix
            int female_id = atoi(female_str + 1);  // Skip 'f' prefix
            mapping[male_id] = female_id;
            count++;
        }
    }
    
    printf("Loaded %d mappings\n", count);
    fclose(file);
    return mapping;
}

// Parallel evaluation functions
void evaluate_permutation_batch(Graph* gm, Graph* gf, int* current_mapping,
                              int* vertices, int** permutations, int batch_size,
                              PermutationResult* results) {
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < batch_size; i++) {
        results[i].delta = calculate_permutation_delta(gm, gf, current_mapping,
                                                     vertices, permutations[i], GROUP_SIZE);
        results[i].permutation = permutations[i];
    }
}

void evaluate_vertex_groups(Graph* gm, Graph* gf, int* current_mapping,
                          VertexGroup* groups, int num_groups,
                          int** permutations, int total_perms) {
    #pragma omp parallel for schedule(dynamic, 1)
    for(int g = 0; g < num_groups; g++) {
        PermutationResult* results = malloc(total_perms * sizeof(PermutationResult));
        evaluate_permutation_batch(gm, gf, current_mapping,
                                 groups[g].vertices, permutations,
                                 total_perms, results);
        
        // Find best result for this group
        groups[g].delta = 0;
        groups[g].best_permutation = malloc(GROUP_SIZE * sizeof(int));
        
        for(int i = 0; i < total_perms; i++) {
            if(results[i].delta > groups[g].delta) {
                groups[g].delta = results[i].delta;
                memcpy(groups[g].best_permutation, results[i].permutation,
                       GROUP_SIZE * sizeof(int));
            }
        }
        
        free(results);
    }
}

void apply_best_permutation(Graph* gm, Graph* gf, int* current_mapping,
                          int* vertices, int* best_perm, int group_size) {
    int* temp_mapping = malloc((NUM_NODES + 1) * sizeof(int));
    memcpy(temp_mapping, current_mapping, (NUM_NODES + 1) * sizeof(int));
    
    for (int i = 0; i < group_size; i++) {
        current_mapping[vertices[i]] = temp_mapping[vertices[best_perm[i]]];
    }
    
    free(temp_mapping);
}

// Helper function to check for duplicates
bool has_duplicate(int* arr, int len, int val) {
    for (int i = 0; i < len; i++) {
        if (arr[i] == val) return true;
    }
    return false;
}

// Main optimization function
// Generate a random permutation
void generate_random_permutation(int* perm, int n) {
    // Initialize with identity permutation
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
}

// Modified function to generate random sample of permutations
int** generate_permutation_samples(int n, int num_samples, int* total_samples) {
    *total_samples = num_samples;
    
    int** permutations = malloc(num_samples * sizeof(int*));
    for(int i = 0; i < num_samples; i++) {
        permutations[i] = malloc(n * sizeof(int));
        generate_random_permutation(permutations[i], n);
        
        // Verify this permutation isn't a duplicate (optional)
        for(int j = 0; j < i; j++) {
            if(memcmp(permutations[i], permutations[j], n * sizeof(int)) == 0) {
                i--; // Try again for this sample
                free(permutations[i+1]);
                break;
            }
        }
    }
    
    return permutations;
}

// Modified optimization function
void optimize_mapping(Graph* gm, Graph* gf, int* current_mapping, const char* out_path) {
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    ProgressStats* stats = init_progress_stats(current_score);
    
    // Generate sample of permutations
    int total_samples;
    int** permutations = generate_permutation_samples(GROUP_SIZE, NUM_SAMPLES, &total_samples);
    
    // Number of vertex groups to evaluate in parallel
    const int NUM_GROUPS = omp_get_max_threads();
    
    printf("\nInitializing optimization with:\n");
    printf("  - Number of threads: %d\n", NUM_GROUPS);
    printf("  - Group size: %d\n", GROUP_SIZE);
    printf("  - Random permutation samples: %d\n", total_samples);
    printf("  - Initial score: %d\n", current_score);
    printf("  - Output path: %s\n\n", out_path);
    
    // Allocate vertex groups
    VertexGroup* groups = malloc(NUM_GROUPS * sizeof(VertexGroup));
    for(int i = 0; i < NUM_GROUPS; i++) {
        groups[i].vertices = malloc(GROUP_SIZE * sizeof(int));
    }
    
    // Initialize random seed
    srand(time(NULL));
    
    while (1) {
        // Generate multiple random vertex groups
        printf("\nIteration %d:\n", stats->total_iterations + 1);
        for(int g = 0; g < NUM_GROUPS; g++) {
            printf("Group %d vertices:", g);
            // Generate unique vertices for this group
            for (int i = 0; i < GROUP_SIZE; i++) {
                int new_vertex;
                do {
                    new_vertex = 1 + rand() % NUM_NODES;
                } while (has_duplicate(groups[g].vertices, i, new_vertex));
                groups[g].vertices[i] = new_vertex;
                printf(" %d", new_vertex);
            }
            printf("\n");
        }
        
        // Evaluate all groups in parallel
        #pragma omp parallel for schedule(dynamic, 1)
        for(int g = 0; g < NUM_GROUPS; g++) {
            PermutationResult* results = malloc(total_samples * sizeof(PermutationResult));
            
            // Evaluate permutations for this group
            #pragma omp parallel for schedule(dynamic, 100)
            for (int i = 0; i < total_samples; i++) {
                results[i].delta = calculate_permutation_delta(gm, gf, current_mapping,
                                                         groups[g].vertices, 
                                                         permutations[i], GROUP_SIZE);
                results[i].permutation = permutations[i];
            }
            
            // Find best result for this group
            groups[g].delta = 0;
            groups[g].best_permutation = malloc(GROUP_SIZE * sizeof(int));
            
            for(int i = 0; i < total_samples; i++) {
                if(results[i].delta > groups[g].delta) {
                    groups[g].delta = results[i].delta;
                    memcpy(groups[g].best_permutation, results[i].permutation,
                           GROUP_SIZE * sizeof(int));
                }
            }
            
            free(results);
        }
        
        // Find best improvement across all groups
        int best_group = -1;
        int best_delta = 0;
        
        for(int g = 0; g < NUM_GROUPS; g++) {
            if(groups[g].delta > best_delta) {
                best_delta = groups[g].delta;
                best_group = g;
            }
        }
        
        // Apply best improvement if found
        if(best_delta > 0) {
            apply_best_permutation(gm, gf, current_mapping,
                                 groups[best_group].vertices,
                                 groups[best_group].best_permutation,
                                 GROUP_SIZE);
            
            // Verify new score
            int new_score = calculate_alignment_score(gm, gf, current_mapping);
            
            update_progress_stats(stats, best_delta, true);
            printf("\nApplied permutation from group %d:\n", best_group);
            printf("  - Delta: %d\n", best_delta);
            printf("  - New score: %d\n", new_score);
            
            // Save immediately with score in filename
            char timestamp_filename[512];
            time_t now = time(NULL);
            struct tm *t = localtime(&now);
            sprintf(timestamp_filename, "%s_score_%d_%04d%02d%02d_%02d%02d%02d.csv", 
                    out_path, new_score,
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                    t->tm_hour, t->tm_min, t->tm_sec);
            
            save_mapping(timestamp_filename, current_mapping, new_score);
        } else {
            update_progress_stats(stats, 0, false);
        }
        
        // Print detailed progress
        if (stats->total_iterations % 100 == 0) {
            print_detailed_progress(stats);
        }
        
        // Clean up this iteration's improvements
        for(int g = 0; g < NUM_GROUPS; g++) {
            if(groups[g].delta > 0) {
                free(groups[g].best_permutation);
            }
        }
    }
    
    // Cleanup
    for(int i = 0; i < total_samples; i++) {
        free(permutations[i]);
    }
    free(permutations);
    
    for(int i = 0; i < NUM_GROUPS; i++) {
        free(groups[i].vertices);
    }
    free(groups);
    free(stats);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <male_graph> <female_graph> <initial_mapping>\n", argv[0]);
        return 1;
    }
    
    // Set number of threads for OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    printf("Starting graph alignment optimization\n");
    printf("Using %d OpenMP threads\n\n", num_threads);
    
    // Load input files
    printf("Loading input files:\n");
    Graph* gm = load_graph_from_csv(argv[1]);
    Graph* gf = load_graph_from_csv(argv[2]);
    int* mapping = load_mapping(argv[3]);
    
    if (!gm || !gf || !mapping) {
        fprintf(stderr, "Error loading input files\n");
        return 1;
    }
    
    // Create output filename
    char output_path[256];
    sprintf(output_path, "improved_%s", argv[3]);
    
    // Run optimization
    printf("\nStarting optimization...\n");
    optimize_mapping(gm, gf, mapping, output_path);
    
    // Cleanup
    printf("\nCleaning up...\n");
    free_graph(gm);
    free_graph(gf);
    free(mapping);
    
    return 0;
}
