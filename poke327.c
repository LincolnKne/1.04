#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h> // For usleep
#include <signal.h> // For signal handling
#include <sys/time.h> // For gettimeofday

#include "heap.h"


#define malloc(size) ({          \
  void *_tmp;                    \
  assert((_tmp = malloc(size))); \
  _tmp;                          \
})

typedef struct path {
  heap_node_t *hn;
  uint8_t pos[2];
  uint8_t from[2];
  int32_t cost;
} path_t;

typedef enum dim {
  DIM_X,
  DIM_Y,
  num_dims
} dim_t;

typedef int16_t pair_t[num_dims];

#define MAP_X              80
#define MAP_Y              21
#define MIN_TREES          10
#define MIN_BOULDERS       10
#define TREE_PROB          95
#define BOULDER_PROB       95
#define WORLD_SIZE         401

#define MOUNTAIN_SYMBOL       '%'
#define BOULDER_SYMBOL        '0'
#define TREE_SYMBOL           '4'
#define FOREST_SYMBOL         '^'
#define GATE_SYMBOL           '#'
#define PATH_SYMBOL           '#'
#define POKEMART_SYMBOL       'M'
#define POKEMON_CENTER_SYMBOL 'C'
#define TALL_GRASS_SYMBOL     ':'
#define SHORT_GRASS_SYMBOL    '.'
#define WATER_SYMBOL          '~'
#define ERROR_SYMBOL          '&'

#define mappair(pair) (m->map[pair[DIM_Y]][pair[DIM_X]])
#define mapxy(x, y) (m->map[y][x])
#define heightpair(pair) (m->height[pair[DIM_Y]][pair[DIM_X]])
#define heightxy(x, y) (m->height[y][x])



typedef enum __attribute__ ((__packed__)) terrain_type {
  ter_boulder,
  ter_tree,
  ter_path,
  ter_mart,
  ter_center,
  ter_grass,
  ter_clearing,
  ter_mountain,
  ter_forest,
  ter_water,
  ter_gate,
  num_terrain_types,
  ter_debug
} terrain_type_t;

typedef enum __attribute__ ((__packed__)) character_type {
  char_pc,
  char_hiker,
  char_rival,
  char_swimmer,
  char_other,
  num_character_types
} character_type_t;

typedef struct pc {
  pair_t pos;
} pc_t;

typedef struct map {
  terrain_type_t map[MAP_Y][MAP_X];
  uint8_t height[MAP_Y][MAP_X];
  int8_t n, s, e, w;
} map_t;

typedef struct queue_node {
  int x, y;
  struct queue_node *next;
} queue_node_t;

#define TOTAL_NUM_TRAINERS 10
#define DIM_X 0
#define DIM_Y 1

typedef struct {
  pair_t gradient;
} gradient_t;

typedef struct npc {
  pair_t pos;  // Position
  char type;   // Type of NPC ('h', 'r', 'p', 'w', 's', 'e')
  int currentDirection[2]; // Current direction for the NPC
  int initialTerrain; // Initial terrain type for the NPC
} npc_t;


typedef struct world {
  map_t *world[WORLD_SIZE][WORLD_SIZE];
  pair_t cur_idx;
  map_t *cur_map;
  /* Place distance maps in world, not map, since *
   * we only need one pair at any given time.     */
  int hiker_dist[MAP_Y][MAP_X];
  int rival_dist[MAP_Y][MAP_X];
  pc_t pc;
  npc_t npcs[TOTAL_NUM_TRAINERS];  // Array of NPCs
} world_t;

/* Even unallocated, a WORLD_SIZE x WORLD_SIZE array of pointers is a very *
 * large thing to put on the stack.  To avoid that, world is a global.     */
world_t world;





/* Just to make the following table fit in 80 columns */
#define IM INT_MAX
int32_t move_cost[num_character_types][num_terrain_types] = {
//  boulder,tree,path,mart,center,grass,clearing,mountain,forest,water,gate
  { IM, IM, 10, 10, 10, 20, 10, IM, IM, IM, 10 },
  { IM, IM, 10, 50, 50, 15, 10, 15, 15, IM, IM },
  { IM, IM, 10, 50, 50, 20, 10, IM, IM, IM, IM },
  { IM, IM, IM, IM, IM, IM, IM, IM, IM,  7, IM },
};
#undef IM



static int32_t path_cmp(const void *key, const void *with) {
  return ((path_t *) key)->cost - ((path_t *) with)->cost;
}

static int32_t edge_penalty(int8_t x, int8_t y)
{
  return (x == 1 || y == 1 || x == MAP_X - 2 || y == MAP_Y - 2) ? 2 : 1;
}

static void dijkstra_path(map_t *m, pair_t from, pair_t to)
{
  static path_t path[MAP_Y][MAP_X], *p;
  static uint32_t initialized = 0;
  heap_t h;
  uint32_t x, y;

  if (!initialized) {
    for (y = 0; y < MAP_Y; y++) {
      for (x = 0; x < MAP_X; x++) {
        path[y][x].pos[DIM_Y] = y;
        path[y][x].pos[DIM_X] = x;
      }
    }
    initialized = 1;
  }
  
  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      path[y][x].cost = INT_MAX;
    }
  }

  path[from[DIM_Y]][from[DIM_X]].cost = 0;

  heap_init(&h, path_cmp, NULL);

  for (y = 1; y < MAP_Y - 1; y++) {
    for (x = 1; x < MAP_X - 1; x++) {
      path[y][x].hn = heap_insert(&h, &path[y][x]);
    }
  }

  while ((p = heap_remove_min(&h))) {
    p->hn = NULL;

    if ((p->pos[DIM_Y] == to[DIM_Y]) && p->pos[DIM_X] == to[DIM_X]) {
      for (x = to[DIM_X], y = to[DIM_Y];
           (x != from[DIM_X]) || (y != from[DIM_Y]);
           p = &path[y][x], x = p->from[DIM_X], y = p->from[DIM_Y]) {
        /* Don't overwrite the gate */
        if (x != to[DIM_X] || y != to[DIM_Y]) {
          mapxy(x, y) = ter_path;
          heightxy(x, y) = 0;
        }
      }
      heap_delete(&h);
      return;
    }

    if ((path[p->pos[DIM_Y] - 1][p->pos[DIM_X]    ].hn) &&
        (path[p->pos[DIM_Y] - 1][p->pos[DIM_X]    ].cost >
         ((p->cost + heightpair(p->pos)) *
          edge_penalty(p->pos[DIM_X], p->pos[DIM_Y] - 1)))) {
      path[p->pos[DIM_Y] - 1][p->pos[DIM_X]    ].cost =
        ((p->cost + heightpair(p->pos)) *
         edge_penalty(p->pos[DIM_X], p->pos[DIM_Y] - 1));
      path[p->pos[DIM_Y] - 1][p->pos[DIM_X]    ].from[DIM_Y] = p->pos[DIM_Y];
      path[p->pos[DIM_Y] - 1][p->pos[DIM_X]    ].from[DIM_X] = p->pos[DIM_X];
      heap_decrease_key_no_replace(&h, path[p->pos[DIM_Y] - 1]
                                           [p->pos[DIM_X]    ].hn);
    }
    if ((path[p->pos[DIM_Y]    ][p->pos[DIM_X] - 1].hn) &&
        (path[p->pos[DIM_Y]    ][p->pos[DIM_X] - 1].cost >
         ((p->cost + heightpair(p->pos)) *
          edge_penalty(p->pos[DIM_X] - 1, p->pos[DIM_Y])))) {
      path[p->pos[DIM_Y]][p->pos[DIM_X] - 1].cost =
        ((p->cost + heightpair(p->pos)) *
         edge_penalty(p->pos[DIM_X] - 1, p->pos[DIM_Y]));
      path[p->pos[DIM_Y]    ][p->pos[DIM_X] - 1].from[DIM_Y] = p->pos[DIM_Y];
      path[p->pos[DIM_Y]    ][p->pos[DIM_X] - 1].from[DIM_X] = p->pos[DIM_X];
      heap_decrease_key_no_replace(&h, path[p->pos[DIM_Y]    ]
                                           [p->pos[DIM_X] - 1].hn);
    }
    if ((path[p->pos[DIM_Y]    ][p->pos[DIM_X] + 1].hn) &&
        (path[p->pos[DIM_Y]    ][p->pos[DIM_X] + 1].cost >
         ((p->cost + heightpair(p->pos)) *
          edge_penalty(p->pos[DIM_X] + 1, p->pos[DIM_Y])))) {
      path[p->pos[DIM_Y]][p->pos[DIM_X] + 1].cost =
        ((p->cost + heightpair(p->pos)) *
         edge_penalty(p->pos[DIM_X] + 1, p->pos[DIM_Y]));
      path[p->pos[DIM_Y]    ][p->pos[DIM_X] + 1].from[DIM_Y] = p->pos[DIM_Y];
      path[p->pos[DIM_Y]    ][p->pos[DIM_X] + 1].from[DIM_X] = p->pos[DIM_X];
      heap_decrease_key_no_replace(&h, path[p->pos[DIM_Y]    ]
                                           [p->pos[DIM_X] + 1].hn);
    }
    if ((path[p->pos[DIM_Y] + 1][p->pos[DIM_X]    ].hn) &&
        (path[p->pos[DIM_Y] + 1][p->pos[DIM_X]    ].cost >
         ((p->cost + heightpair(p->pos)) *
          edge_penalty(p->pos[DIM_X], p->pos[DIM_Y] + 1)))) {
      path[p->pos[DIM_Y] + 1][p->pos[DIM_X]    ].cost =
        ((p->cost + heightpair(p->pos)) *
         edge_penalty(p->pos[DIM_X], p->pos[DIM_Y] + 1));
      path[p->pos[DIM_Y] + 1][p->pos[DIM_X]    ].from[DIM_Y] = p->pos[DIM_Y];
      path[p->pos[DIM_Y] + 1][p->pos[DIM_X]    ].from[DIM_X] = p->pos[DIM_X];
      heap_decrease_key_no_replace(&h, path[p->pos[DIM_Y] + 1]
                                           [p->pos[DIM_X]    ].hn);
    }
  }
}
int compare_characters(const void *a, const void *b) {
  const npc_t *npc_a = (const npc_t *)a;
  const npc_t *npc_b = (const npc_t *)b;
  return npc_a->type - npc_b->type;
}

void insert_character_into_heap(heap_t *heap, char type) {
  npc_t *new_npc = malloc(sizeof(npc_t));
  if (new_npc == NULL) {
    fprintf(stderr, "Memory allocation failed\\n");
    exit(EXIT_FAILURE);
  }
  new_npc->type = type;
  heap_insert(heap, new_npc);
}


static int build_paths(map_t *m)
{
  pair_t from, to;

  /*  printf("%d %d %d %d\n", m->n, m->s, m->e, m->w);*/

  if (m->e != -1 && m->w != -1) {
    from[DIM_X] = 1;
    to[DIM_X] = MAP_X - 2;
    from[DIM_Y] = m->w;
    to[DIM_Y] = m->e;

    dijkstra_path(m, from, to);
  }

  if (m->n != -1 && m->s != -1) {
    from[DIM_Y] = 1;
    to[DIM_Y] = MAP_Y - 2;
    from[DIM_X] = m->n;
    to[DIM_X] = m->s;

    dijkstra_path(m, from, to);
  }

  if (m->e == -1) {
    if (m->s == -1) {
      from[DIM_X] = 1;
      from[DIM_Y] = m->w;
      to[DIM_X] = m->n;
      to[DIM_Y] = 1;
    } else {
      from[DIM_X] = 1;
      from[DIM_Y] = m->w;
      to[DIM_X] = m->s;
      to[DIM_Y] = MAP_Y - 2;
    }

    dijkstra_path(m, from, to);
  }

  if (m->w == -1) {
    if (m->s == -1) {
      from[DIM_X] = MAP_X - 2;
      from[DIM_Y] = m->e;
      to[DIM_X] = m->n;
      to[DIM_Y] = 1;
    } else {
      from[DIM_X] = MAP_X - 2;
      from[DIM_Y] = m->e;
      to[DIM_X] = m->s;
      to[DIM_Y] = MAP_Y - 2;
    }

    dijkstra_path(m, from, to);
  }

  if (m->n == -1) {
    if (m->e == -1) {
      from[DIM_X] = 1;
      from[DIM_Y] = m->w;
      to[DIM_X] = m->s;
      to[DIM_Y] = MAP_Y - 2;
    } else {
      from[DIM_X] = MAP_X - 2;
      from[DIM_Y] = m->e;
      to[DIM_X] = m->s;
      to[DIM_Y] = MAP_Y - 2;
    }

    dijkstra_path(m, from, to);
  }

  if (m->s == -1) {
    if (m->e == -1) {
      from[DIM_X] = 1;
      from[DIM_Y] = m->w;
      to[DIM_X] = m->n;
      to[DIM_Y] = 1;
    } else {
      from[DIM_X] = MAP_X - 2;
      from[DIM_Y] = m->e;
      to[DIM_X] = m->n;
      to[DIM_Y] = 1;
    }

    dijkstra_path(m, from, to);
  }

  return 0;
}

static int gaussian[5][5] = {
  {  1,  4,  7,  4,  1 },
  {  4, 16, 26, 16,  4 },
  {  7, 26, 41, 26,  7 },
  {  4, 16, 26, 16,  4 },
  {  1,  4,  7,  4,  1 }
};

static int smooth_height(map_t *m)
{
  int32_t i, x, y;
  int32_t s, t, p, q;
  queue_node_t *head, *tail, *tmp;
  /*  FILE *out;*/
  uint8_t height[MAP_Y][MAP_X];

  memset(&height, 0, sizeof (height));

  /* Seed with some values */
  for (i = 1; i < 255; i += 20) {
    do {
      x = rand() % MAP_X;
      y = rand() % MAP_Y;
    } while (height[y][x]);
    height[y][x] = i;
    if (i == 1) {
      head = tail = malloc(sizeof (*tail));
    } else {
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
    }
    tail->next = NULL;
    tail->x = x;
    tail->y = y;
  }

  /*
  out = fopen("seeded.pgm", "w");
  fprintf(out, "P5\n%u %u\n255\n", MAP_X, MAP_Y);
  fwrite(&height, sizeof (height), 1, out);
  fclose(out);
  */
  
  /* Diffuse the vaules to fill the space */
  while (head) {
    x = head->x;
    y = head->y;
    i = height[y][x];

    if (x - 1 >= 0 && y - 1 >= 0 && !height[y - 1][x - 1]) {
      height[y - 1][x - 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x - 1;
      tail->y = y - 1;
    }
    if (x - 1 >= 0 && !height[y][x - 1]) {
      height[y][x - 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x - 1;
      tail->y = y;
    }
    if (x - 1 >= 0 && y + 1 < MAP_Y && !height[y + 1][x - 1]) {
      height[y + 1][x - 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x - 1;
      tail->y = y + 1;
    }
    if (y - 1 >= 0 && !height[y - 1][x]) {
      height[y - 1][x] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x;
      tail->y = y - 1;
    }
    if (y + 1 < MAP_Y && !height[y + 1][x]) {
      height[y + 1][x] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x;
      tail->y = y + 1;
    }
    if (x + 1 < MAP_X && y - 1 >= 0 && !height[y - 1][x + 1]) {
      height[y - 1][x + 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x + 1;
      tail->y = y - 1;
    }
    if (x + 1 < MAP_X && !height[y][x + 1]) {
      height[y][x + 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x + 1;
      tail->y = y;
    }
    if (x + 1 < MAP_X && y + 1 < MAP_Y && !height[y + 1][x + 1]) {
      height[y + 1][x + 1] = i;
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
      tail->next = NULL;
      tail->x = x + 1;
      tail->y = y + 1;
    }

    tmp = head;
    head = head->next;
    free(tmp);
  }

  /* And smooth it a bit with a gaussian convolution */
  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      for (s = t = p = 0; p < 5; p++) {
        for (q = 0; q < 5; q++) {
          if (y + (p - 2) >= 0 && y + (p - 2) < MAP_Y &&
              x + (q - 2) >= 0 && x + (q - 2) < MAP_X) {
            s += gaussian[p][q];
            t += height[y + (p - 2)][x + (q - 2)] * gaussian[p][q];
          }
        }
      }
      m->height[y][x] = t / s;
    }
  }
  /* Let's do it again, until it's smooth like Kenny G. */
  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      for (s = t = p = 0; p < 5; p++) {
        for (q = 0; q < 5; q++) {
          if (y + (p - 2) >= 0 && y + (p - 2) < MAP_Y &&
              x + (q - 2) >= 0 && x + (q - 2) < MAP_X) {
            s += gaussian[p][q];
            t += height[y + (p - 2)][x + (q - 2)] * gaussian[p][q];
          }
        }
      }
      m->height[y][x] = t / s;
    }
  }

  /*
  out = fopen("diffused.pgm", "w");
  fprintf(out, "P5\n%u %u\n255\n", MAP_X, MAP_Y);
  fwrite(&height, sizeof (height), 1, out);
  fclose(out);

  out = fopen("smoothed.pgm", "w");
  fprintf(out, "P5\n%u %u\n255\n", MAP_X, MAP_Y);
  fwrite(&m->height, sizeof (m->height), 1, out);
  fclose(out);
  */

  return 0;
}

static void find_building_location(map_t *m, pair_t p)
{
  do {
    p[DIM_X] = rand() % (MAP_X - 3) + 1;
    p[DIM_Y] = rand() % (MAP_Y - 3) + 1;

    if ((((mapxy(p[DIM_X] - 1, p[DIM_Y]    ) == ter_path)     &&
          (mapxy(p[DIM_X] - 1, p[DIM_Y] + 1) == ter_path))    ||
         ((mapxy(p[DIM_X] + 2, p[DIM_Y]    ) == ter_path)     &&
          (mapxy(p[DIM_X] + 2, p[DIM_Y] + 1) == ter_path))    ||
         ((mapxy(p[DIM_X]    , p[DIM_Y] - 1) == ter_path)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y] - 1) == ter_path))    ||
         ((mapxy(p[DIM_X]    , p[DIM_Y] + 2) == ter_path)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y] + 2) == ter_path)))   &&
        (((mapxy(p[DIM_X]    , p[DIM_Y]    ) != ter_mart)     &&
          (mapxy(p[DIM_X]    , p[DIM_Y]    ) != ter_center)   &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y]    ) != ter_mart)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y]    ) != ter_center)   &&
          (mapxy(p[DIM_X]    , p[DIM_Y] + 1) != ter_mart)     &&
          (mapxy(p[DIM_X]    , p[DIM_Y] + 1) != ter_center)   &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y] + 1) != ter_mart)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y] + 1) != ter_center))) &&
        (((mapxy(p[DIM_X]    , p[DIM_Y]    ) != ter_path)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y]    ) != ter_path)     &&
          (mapxy(p[DIM_X]    , p[DIM_Y] + 1) != ter_path)     &&
          (mapxy(p[DIM_X] + 1, p[DIM_Y] + 1) != ter_path)))) {
          break;
    }
  } while (1);
}

static int place_pokemart(map_t *m)
{
  pair_t p;

  find_building_location(m, p);

  mapxy(p[DIM_X]    , p[DIM_Y]    ) = ter_mart;
  mapxy(p[DIM_X] + 1, p[DIM_Y]    ) = ter_mart;
  mapxy(p[DIM_X]    , p[DIM_Y] + 1) = ter_mart;
  mapxy(p[DIM_X] + 1, p[DIM_Y] + 1) = ter_mart;

  return 0;
}

static int place_center(map_t *m)
{  pair_t p;

  find_building_location(m, p);

  mapxy(p[DIM_X]    , p[DIM_Y]    ) = ter_center;
  mapxy(p[DIM_X] + 1, p[DIM_Y]    ) = ter_center;
  mapxy(p[DIM_X]    , p[DIM_Y] + 1) = ter_center;
  mapxy(p[DIM_X] + 1, p[DIM_Y] + 1) = ter_center;

  return 0;
}

/* Chooses tree or boulder for border cell.  Choice is biased by dominance *
 * of neighboring cells.                                                   */
static terrain_type_t border_type(map_t *m, int32_t x, int32_t y)
{
  int32_t p, q;
  int32_t r, t;
  int32_t miny, minx, maxy, maxx;
  
  r = t = 0;
  
  miny = y - 1 >= 0 ? y - 1 : 0;
  maxy = y + 1 <= MAP_Y ? y + 1: MAP_Y;
  minx = x - 1 >= 0 ? x - 1 : 0;
  maxx = x + 1 <= MAP_X ? x + 1: MAP_X;

  for (q = miny; q < maxy; q++) {
    for (p = minx; p < maxx; p++) {
      if (q != y || p != x) {
        if (m->map[q][p] == ter_mountain ||
            m->map[q][p] == ter_boulder) {
          r++;
        } else if (m->map[q][p] == ter_forest ||
                   m->map[q][p] == ter_tree) {
          t++;
        }
      }
    }
  }
  
  if (t == r) {
    return rand() & 1 ? ter_boulder : ter_tree;
  } else if (t > r) {
    if (rand() % 10) {
      return ter_tree;
    } else {
      return ter_boulder;
    }
  } else {
    if (rand() % 10) {
      return ter_boulder;
    } else {
      return ter_tree;
    }
  }
}

static int map_terrain(map_t *m, int8_t n, int8_t s, int8_t e, int8_t w)
{
  int32_t i, x, y;
  queue_node_t *head, *tail, *tmp;
  //  FILE *out;
  int num_grass, num_clearing, num_mountain, num_forest, num_water, num_total;
  terrain_type_t type;
  int added_current = 0;
  
  num_grass = rand() % 4 + 2;
  num_clearing = rand() % 4 + 2;
  num_mountain = rand() % 2 + 1;
  num_forest = rand() % 2 + 1;
  num_water = rand() % 2 + 1;
  num_total = num_grass + num_clearing + num_mountain + num_forest + num_water;

  memset(&m->map, 0, sizeof (m->map));

  /* Seed with some values */
  for (i = 0; i < num_total; i++) {
    do {
      x = rand() % MAP_X;
      y = rand() % MAP_Y;
    } while (m->map[y][x]);
    if (i == 0) {
      type = ter_grass;
    } else if (i == num_grass) {
      type = ter_clearing;
    } else if (i == num_grass + num_clearing) {
      type = ter_mountain;
    } else if (i == num_grass + num_clearing + num_mountain) {
      type = ter_forest;
    } else if (i == num_grass + num_clearing + num_mountain + num_forest) {
      type = ter_water;
    }
    m->map[y][x] = type;
    if (i == 0) {
      head = tail = malloc(sizeof (*tail));
    } else {
      tail->next = malloc(sizeof (*tail));
      tail = tail->next;
    }
    tail->next = NULL;
    tail->x = x;
    tail->y = y;
  }

  /*
  out = fopen("seeded.pgm", "w");
  fprintf(out, "P5\n%u %u\n255\n", MAP_X, MAP_Y);
  fwrite(&m->map, sizeof (m->map), 1, out);
  fclose(out);
  */

  /* Diffuse the vaules to fill the space */
  while (head) {
    x = head->x;
    y = head->y;
    i = m->map[y][x];
    
    if (x - 1 >= 0 && !m->map[y][x - 1]) {
      if ((rand() % 100) < 80) {
        m->map[y][x - 1] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x - 1;
        tail->y = y;
      } else if (!added_current) {
        added_current = 1;
        m->map[y][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y;
      }
    }

    if (y - 1 >= 0 && !m->map[y - 1][x]) {
      if ((rand() % 100) < 20) {
        m->map[y - 1][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y - 1;
      } else if (!added_current) {
        added_current = 1;
        m->map[y][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y;
      }
    }

    if (y + 1 < MAP_Y && !m->map[y + 1][x]) {
      if ((rand() % 100) < 20) {
        m->map[y + 1][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y + 1;
      } else if (!added_current) {
        added_current = 1;
        m->map[y][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y;
      }
    }

    if (x + 1 < MAP_X && !m->map[y][x + 1]) {
      if ((rand() % 100) < 80) {
        m->map[y][x + 1] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x + 1;
        tail->y = y;
      } else if (!added_current) {
        added_current = 1;
        m->map[y][x] = i;
        tail->next = malloc(sizeof (*tail));
        tail = tail->next;
        tail->next = NULL;
        tail->x = x;
        tail->y = y;
      }
    }

    added_current = 0;
    tmp = head;
    head = head->next;
    free(tmp);
  }

  /*
  out = fopen("diffused.pgm", "w");
  fprintf(out, "P5\n%u %u\n255\n", MAP_X, MAP_Y);
  fwrite(&m->map, sizeof (m->map), 1, out);
  fclose(out);
  */
  
  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      if (y == 0 || y == MAP_Y - 1 ||
          x == 0 || x == MAP_X - 1) {
        mapxy(x, y) = border_type(m, x, y);
      }
    }
  }

  m->n = n;
  m->s = s;
  m->e = e;
  m->w = w;

  if (n != -1) {
    mapxy(n,         0        ) = ter_gate;
    mapxy(n,         1        ) = ter_gate;
  }
  if (s != -1) {
    mapxy(s,         MAP_Y - 1) = ter_gate;
    mapxy(s,         MAP_Y - 2) = ter_gate;
  }
  if (w != -1) {
    mapxy(0,         w        ) = ter_gate;
    mapxy(1,         w        ) = ter_gate;
  }
  if (e != -1) {
    mapxy(MAP_X - 1, e        ) = ter_gate;
    mapxy(MAP_X - 2, e        ) = ter_gate;
  }

  return 0;
}

static int place_boulders(map_t *m)
{
  int i;
  int x, y;

  for (i = 0; i < MIN_BOULDERS || rand() % 100 < BOULDER_PROB; i++) {
    y = rand() % (MAP_Y - 2) + 1;
    x = rand() % (MAP_X - 2) + 1;
    if (m->map[y][x] != ter_forest &&
        m->map[y][x] != ter_path   &&
        m->map[y][x] != ter_gate) {
      m->map[y][x] = ter_boulder;
    }
  }

  return 0;
}

static int place_trees(map_t *m)
{
  int i;
  int x, y;
  
  for (i = 0; i < MIN_TREES || rand() % 100 < TREE_PROB; i++) {
    y = rand() % (MAP_Y - 2) + 1;
    x = rand() % (MAP_X - 2) + 1;
    if (m->map[y][x] != ter_mountain &&
        m->map[y][x] != ter_path     &&
        m->map[y][x] != ter_water    &&
        m->map[y][x] != ter_gate) {
      m->map[y][x] = ter_tree;
    }
  }

  return 0;
}

// New map expects cur_idx to refer to the index to be generated.  If that
// map has already been generated then the only thing this does is set
// cur_map.
static int new_map()
{
  int d, p;
  int e, w, n, s;

  if (world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X]]) {
    world.cur_map = world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X]];
    return 0;
  }

  world.cur_map                                             =
    world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X]] =
    malloc(sizeof (*world.cur_map));

  smooth_height(world.cur_map);
  
  if (!world.cur_idx[DIM_Y]) {
    n = -1;
  } else if (world.world[world.cur_idx[DIM_Y] - 1][world.cur_idx[DIM_X]]) {
    n = world.world[world.cur_idx[DIM_Y] - 1][world.cur_idx[DIM_X]]->s;
  } else {
    n = 1 + rand() % (MAP_X - 2);
  }
  if (world.cur_idx[DIM_Y] == WORLD_SIZE - 1) {
    s = -1;
  } else if (world.world[world.cur_idx[DIM_Y] + 1][world.cur_idx[DIM_X]]) {
    s = world.world[world.cur_idx[DIM_Y] + 1][world.cur_idx[DIM_X]]->n;
  } else  {
    s = 1 + rand() % (MAP_X - 2);
  }
  if (!world.cur_idx[DIM_X]) {
    w = -1;
  } else if (world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X] - 1]) {
    w = world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X] - 1]->e;
  } else {
    w = 1 + rand() % (MAP_Y - 2);
  }
  if (world.cur_idx[DIM_X] == WORLD_SIZE - 1) {
    e = -1;
  } else if (world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X] + 1]) {
    e = world.world[world.cur_idx[DIM_Y]][world.cur_idx[DIM_X] + 1]->w;
  } else {
    e = 1 + rand() % (MAP_Y - 2);
  }
  
  map_terrain(world.cur_map, n, s, e, w);
     
  place_boulders(world.cur_map);
  place_trees(world.cur_map);
  build_paths(world.cur_map);
  d = (abs(world.cur_idx[DIM_X] - (WORLD_SIZE / 2)) +
       abs(world.cur_idx[DIM_Y] - (WORLD_SIZE / 2)));
  p = d > 200 ? 5 : (50 - ((45 * d) / 200));
  //  printf("d=%d, p=%d\n", d, p);
  if ((rand() % 100) < p || !d) {
    place_pokemart(world.cur_map);
  }
  if ((rand() % 100) < p || !d) {
    place_center(world.cur_map);
  }

  return 0;
}

static void print_map()
{
  printf("\033[H\033[J");
  int x, y;
  int default_reached = 0;
  char npc_symbol; // Variable to hold the NPC symbol at a given position

  printf("\n\n\n");

  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      npc_symbol = 0; // Reset the NPC symbol for each cell

      // Check if an NPC is at this position
      for (int i = 0; i < TOTAL_NUM_TRAINERS; i++) {
        if (world.npcs[i].pos[DIM_Y] == y && world.npcs[i].pos[DIM_X] == x) {
          npc_symbol = world.npcs[i].type;
          break; // Stop searching if we find an NPC
        }
      }

      if (world.pc.pos[DIM_Y] == y && world.pc.pos[DIM_X] == x) {
        putchar('@');
      } else if (npc_symbol != 0) {
        putchar(npc_symbol);  // Draw NPC if present
      } else {
        switch (world.cur_map->map[y][x]) {
        case ter_boulder:
          putchar(BOULDER_SYMBOL);
          break;
        case ter_mountain:
          putchar(MOUNTAIN_SYMBOL);
          break;
        case ter_tree:
          putchar(TREE_SYMBOL);
          break;
        case ter_forest:
          putchar(FOREST_SYMBOL);
          break;
        case ter_path:
          putchar(PATH_SYMBOL);
          break;
        case ter_gate:
          putchar(GATE_SYMBOL);
          break;
        case ter_mart:
          putchar(POKEMART_SYMBOL);
          break;
        case ter_center:
          putchar(POKEMON_CENTER_SYMBOL);
          break;
        case ter_grass:
          putchar(TALL_GRASS_SYMBOL);
          break;
        case ter_clearing:
          putchar(SHORT_GRASS_SYMBOL);
          break;
        case ter_water:
          putchar(WATER_SYMBOL);
          break;
        default:
          putchar(ERROR_SYMBOL);
          default_reached = 1;
          break;
        }
      }
    }
    putchar('\n');
  }

  if (default_reached) {
    fprintf(stderr, "Default reached in %s\n", __FUNCTION__);
  }
}

// The world is global because of its size, so init_world is parameterless
void init_world()
{
  world.cur_idx[DIM_X] = world.cur_idx[DIM_Y] = WORLD_SIZE / 2;
  new_map();
}

void delete_world()
{
  int x, y;

  for (y = 0; y < WORLD_SIZE; y++) {
    for (x = 0; x < WORLD_SIZE; x++) {
      if (world.world[y][x]) {
        free(world.world[y][x]);
        world.world[y][x] = NULL;
      }
    }
  }
}

#define ter_cost(x, y, c) move_cost[c][m->map[y][x]]

static int32_t hiker_cmp(const void *key, const void *with) {
  return (world.hiker_dist[((path_t *) key)->pos[DIM_Y]]
                          [((path_t *) key)->pos[DIM_X]] -
          world.hiker_dist[((path_t *) with)->pos[DIM_Y]]
                          [((path_t *) with)->pos[DIM_X]]);
}

static int32_t rival_cmp(const void *key, const void *with) {
  return (world.rival_dist[((path_t *) key)->pos[DIM_Y]]
                          [((path_t *) key)->pos[DIM_X]] -
          world.rival_dist[((path_t *) with)->pos[DIM_Y]]
                          [((path_t *) with)->pos[DIM_X]]);
}

void pathfind(map_t *m)
{
  heap_t h;
  uint32_t x, y;
  static path_t p[MAP_Y][MAP_X], *c;
  static uint32_t initialized = 0;

  if (!initialized) {
    initialized = 1;
    for (y = 0; y < MAP_Y; y++) {
      for (x = 0; x < MAP_X; x++) {
        p[y][x].pos[DIM_Y] = y;
        p[y][x].pos[DIM_X] = x;
      }
    }
  }

  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      world.hiker_dist[y][x] = world.rival_dist[y][x] = INT_MAX;
    }
  }
  world.hiker_dist[world.pc.pos[DIM_Y]][world.pc.pos[DIM_X]] = 
    world.rival_dist[world.pc.pos[DIM_Y]][world.pc.pos[DIM_X]] = 0;

  heap_init(&h, hiker_cmp, NULL);

  for (y = 1; y < MAP_Y - 1; y++) {
    for (x = 1; x < MAP_X - 1; x++) {
      if (ter_cost(x, y, char_hiker) != INT_MAX) {
        p[y][x].hn = heap_insert(&h, &p[y][x]);
      } else {
        p[y][x].hn = NULL;
      }
    }
  }

  while ((c = heap_remove_min(&h))) {
    c->hn = NULL;
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ].hn);
    }
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1].hn);
    }
    if ((p[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1].hn) &&
        (world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1] >
         world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker))) {
      world.hiker_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1] =
        world.hiker_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_hiker);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1].hn);
    }
  }
  heap_delete(&h);

  heap_init(&h, rival_cmp, NULL);

  for (y = 1; y < MAP_Y - 1; y++) {
    for (x = 1; x < MAP_X - 1; x++) {
      if (ter_cost(x, y, char_rival) != INT_MAX) {
        p[y][x].hn = heap_insert(&h, &p[y][x]);
      } else {
        p[y][x].hn = NULL;
      }
    }
  }

  while ((c = heap_remove_min(&h))) {
    c->hn = NULL;
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ].hn) &&
        (world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X]    ].hn);
    }
    if ((p[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] - 1][c->pos[DIM_X] + 1].hn);
    }
    if ((p[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y]    ][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y]    ][c->pos[DIM_X] + 1].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X] - 1].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ].hn) &&
        (world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X]    ].hn);
    }
    if ((p[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1].hn) &&
        (world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1] >
         world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
         ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival))) {
      world.rival_dist[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1] =
        world.rival_dist[c->pos[DIM_Y]][c->pos[DIM_X]] +
        ter_cost(c->pos[DIM_X], c->pos[DIM_Y], char_rival);
      heap_decrease_key_no_replace(&h,
                                   p[c->pos[DIM_Y] + 1][c->pos[DIM_X] + 1].hn);
    }
  }
  heap_delete(&h);
}

void init_pc()
{
  int x, y;

  do {
    x = rand() % (MAP_X - 2) + 1;
    y = rand() % (MAP_Y - 2) + 1;
  } while (world.cur_map->map[y][x] != ter_path);

  world.pc.pos[DIM_X] = x;
  world.pc.pos[DIM_Y] = y;
}

void init_npcs() {
  int x, y, i;
  int remaining_trainers = TOTAL_NUM_TRAINERS; // Local variable to keep track
  char types[] = {'h', 'r', 'p', 'w', 's', 'e'};
  
  // Ensure at least one hiker and one rival
  if (remaining_trainers >= 2) {
    world.npcs[0].type = 'h';
    world.npcs[1].type = 'r';
    remaining_trainers -= 2; // Update the local variable, not TOTAL_NUM_TRAINERS
  }
  
  // Initialize other NPCs
  for (i = 2; i < TOTAL_NUM_TRAINERS; ++i) {
    if (i < remaining_trainers + 2) { // +2 because we already initialized 2 NPCs
      world.npcs[i].type = types[rand() % 6];
    }
  }
  
  // Set positions
  for (i = 0; i < TOTAL_NUM_TRAINERS; ++i) {
    do {
      x = rand() % (MAP_X - 2) + 1;
      y = rand() % (MAP_Y - 2) + 1;
    } while (world.cur_map->map[y][x] != ter_path);
    
    world.npcs[i].pos[DIM_X] = x;
    world.npcs[i].pos[DIM_Y] = y;
  }
}

character_type_t mapCharToEnum(char type) {
  switch(type) {
    case 'h': return char_hiker;
    case 'r': return char_rival;
    case 's': return char_swimmer;
    default: return char_other;
  }
}

int isValidPosition(int x, int y) {
  // Check if the position is within the map bounds
  if (x < 0 || x >= MAP_X || y < 0 || y >= MAP_Y) {
    return 0; // False
  }

  // Check if the terrain is passable
  int terrain = world.cur_map->map[y][x];
  if (terrain == ter_boulder || terrain == ter_mountain || terrain == ter_water) {
    return 0; // False
  }

  return 1; // True
}

int isCellOccupied(int x, int y, npc_t npcs[]) {
  for (int i = 0; i < TOTAL_NUM_TRAINERS; i++) {
    if (npcs[i].pos[DIM_X] == x && npcs[i].pos[DIM_Y] == y) {
      return 1; // True, cell is occupied
    }
  }
  return 0; // False, cell is not occupied
}


gradient_t getGradientForHiker(int x, int y, int hiker_dist[MAP_Y][MAP_X], terrain_type_t terrain_map[MAP_Y][MAP_X], npc_t *npc);
gradient_t getGradientForRival(int x, int y, int rival_dist[MAP_Y][MAP_X], terrain_type_t terrain_map[MAP_Y][MAP_X], npc_t *npc);


gradient_t getGradientForHiker(int x, int y, int hiker_dist[MAP_Y][MAP_X], terrain_type_t terrain_map[MAP_Y][MAP_X], npc_t *npc){
  gradient_t gradient;
  int min_val = INT_MAX;
  int dx, dy;
  character_type_t mappedType = mapCharToEnum(npc->type);

  for (dy = -1; dy <= 1; dy++) {
    for (dx = -1; dx <= 1; dx++) {
      int new_x = x + dx;
      int new_y = y + dy;

      if (isValidPosition(new_x, new_y)) {
        int terrain = terrain_map[new_y][new_x];
        if (hiker_dist[new_y][new_x] != INT_MAX && move_cost[mappedType][terrain] != INT_MAX) {
          if (hiker_dist[new_y][new_x] < min_val) {
            min_val = hiker_dist[new_y][new_x];
            gradient.gradient[DIM_X] = dx;
            gradient.gradient[DIM_Y] = dy;
          }
        }
      }
    }
  }

  return gradient;
}

gradient_t getGradientForRival(int x, int y, int rival_dist[MAP_Y][MAP_X], terrain_type_t terrain_map[MAP_Y][MAP_X], npc_t *npc){
  gradient_t gradient;
  int min_val = INT_MAX;
  int dx, dy;
  character_type_t mappedType = mapCharToEnum(npc->type);

  for (dy = -1; dy <= 1; dy++) {
    for (dx = -1; dx <= 1; dx++) {
      int new_x = x + dx;
      int new_y = y + dy;

      if (isValidPosition(new_x, new_y)) {
        int terrain = terrain_map[new_y][new_x];
        if (rival_dist[new_y][new_x] != INT_MAX && move_cost[mappedType][terrain] != INT_MAX) {
          if (rival_dist[new_y][new_x] < min_val) {
            min_val = rival_dist[new_y][new_x];
            gradient.gradient[DIM_X] = dx;
            gradient.gradient[DIM_Y] = dy;
          }
        }
      }
    }
  }

  return gradient;
}



void move_npc(npc_t *npc, npc_t npcs[]) { 
  int dx = 0, dy = 0;
  int newX, newY;
  gradient_t gradient;

  switch (npc->type) {
    case 'h':
      gradient = getGradientForHiker(npc->pos[DIM_X], npc->pos[DIM_Y], world.hiker_dist, world.cur_map->map, npc);
      dx = gradient.gradient[DIM_X];
      dy = gradient.gradient[DIM_Y];
      break;
    case 'r':
      gradient = getGradientForRival(npc->pos[DIM_X], npc->pos[DIM_Y], world.rival_dist, world.cur_map->map, npc);
      dx = gradient.gradient[DIM_X];
      dy = gradient.gradient[DIM_Y];
      break;
    case 'p':
      dx = npc->currentDirection[0];  // Assuming currentDirection is an array
      if (world.cur_map->map[npc->pos[DIM_Y]][npc->pos[DIM_X] + dx] == INT_MAX) {
        npc->currentDirection[0] *= -1;  // Reverse direction
      }
      break;

    case 'w': // Wanderer
      dx = npc->currentDirection[DIM_X];
      dy = npc->currentDirection[DIM_Y];
      if (world.cur_map->map[npc->pos[DIM_Y] + dy][npc->pos[DIM_X] + dx] != npc->initialTerrain) {
        // Change direction randomly
        dx = rand() % 3 - 1;
        dy = rand() % 3 - 1;
      }
      break;

    case 's': // Sentry
      // Sentries don't move
      return;

    case 'e': // Explorer
      dx = npc->currentDirection[DIM_X];
      dy = npc->currentDirection[DIM_Y];
      if (world.cur_map->map[npc->pos[DIM_Y] + dy][npc->pos[DIM_X] + dx] == INT_MAX) {
        // Change direction randomly
        dx = rand() % 3 - 1;
        dy = rand() % 3 - 1;
      }
      break;
  }

    newX = npc->pos[DIM_X] + dx;
  newY = npc->pos[DIM_Y] + dy;

  if (isValidPosition(newX, newY) && !isCellOccupied(newX, newY, npcs)) {  
    npc->pos[DIM_X] = newX;
    npc->pos[DIM_Y] = newY;
  }
}

void print_hiker_dist()
{
  int x, y;

  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      if (world.hiker_dist[y][x] == INT_MAX) {
        printf("   ");
      } else {
        printf(" %02d", world.hiker_dist[y][x] % 100);
      }
    }
    printf("\n");
  }
}

void print_rival_dist()
{
  int x, y;

  for (y = 0; y < MAP_Y; y++) {
    for (x = 0; x < MAP_X; x++) {
      if (world.rival_dist[y][x] == INT_MAX || world.rival_dist[y][x] < 0) {
        printf("   ");
      } else {
        printf(" %02d", world.rival_dist[y][x] % 100);
      }
    }
    printf("\n");
  }
}
volatile sig_atomic_t flag = 0;

void handle_sigint(int sig) {
  flag = 1;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#include <stdint.h>


int main(int argc, char *argv[]) {
  // Initialize variables
  struct timeval tv;
  uint32_t seed;
  char c;
  int x, y;

  // Initialize totalNumTrainers with the value of TOTAL_NUM_TRAINERS
  int totalNumTrainers = TOTAL_NUM_TRAINERS;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--TOTAL_NUM_TRAINERS") == 0) {
      if (i + 1 < argc) {
        totalNumTrainers = atoi(argv[i + 1]);  // Update totalNumTrainers
        i++;
      } else {
        fprintf(stderr, "Option --TOTAL_NUM_TRAINERS requires an argument.\n");
        exit(EXIT_FAILURE);
      }
    }
  }

  // Seed initialization
  if (argc == 2) {
    seed = atoi(argv[1]);
  } else {
    gettimeofday(&tv, NULL);
    seed = (tv.tv_usec ^ (tv.tv_sec << 20)) & 0xffffffff;
  }

  // Initialize character heap
  heap_t character_heap;
  heap_init(&character_heap, compare_characters, NULL);

  // Insert characters into heap
  for (int i = 0; i < totalNumTrainers; i++) {  // Use totalNumTrainers here
    insert_character_into_heap(&character_heap, 'h'); // char_hiker
    insert_character_into_heap(&character_heap, 'r'); // char_rival
    insert_character_into_heap(&character_heap, 'p'); // Pacers
    insert_character_into_heap(&character_heap, 'w'); // Wanderers
    insert_character_into_heap(&character_heap, 's'); // Sentries
    insert_character_into_heap(&character_heap, 'e'); // Explorers
  }

  // Seed and world initialization
  printf("Using seed: %u\n", seed);
  srand(seed);
  init_world();
  init_pc();
  init_npcs();
  pathfind(world.cur_map);

  // Register the signal handler for Ctrl+C
  signal(SIGINT, handle_sigint);

  // Main game loop
  do {
    if (flag) {
      printf("Game terminated using Ctrl+C.\n");
      break;
    }
    // Move NPCs
  for (int i = 0; i < TOTAL_NUM_TRAINERS; ++i) {
    move_npc(&world.npcs[i], world.npcs);
  }
    // Redraw the map
    print_map();

    // Display current position
    printf("Current position is %d%cx%d%c (%d,%d).  "
           "Enter command: ",
           abs(world.cur_idx[DIM_X] - (WORLD_SIZE / 2)),
           world.cur_idx[DIM_X] - (WORLD_SIZE / 2) >= 0 ? 'E' : 'W',
           abs(world.cur_idx[DIM_Y] - (WORLD_SIZE / 2)),
           world.cur_idx[DIM_Y] - (WORLD_SIZE / 2) <= 0 ? 'N' : 'S',
           world.cur_idx[DIM_X] - (WORLD_SIZE / 2),
           world.cur_idx[DIM_Y] - (WORLD_SIZE / 2));

    if (scanf(" %c", &c) != 1) {
      putchar('\n');
      break;
    }

    // Handle user input
    switch (c) {
      case 'n':
        // Move north
        if (world.cur_idx[DIM_Y]) {
          world.cur_idx[DIM_Y]--;
          new_map();
        }
        break;
      case 's':
        // Move south
        if (world.cur_idx[DIM_Y] < WORLD_SIZE - 1) {
          world.cur_idx[DIM_Y]++;
          new_map();
        }
        break;
      case 'e':
        // Move east
        if (world.cur_idx[DIM_X] < WORLD_SIZE - 1) {
          world.cur_idx[DIM_X]++;
          new_map();
        }
        break;
      case 'w':
        // Move west
        if (world.cur_idx[DIM_X]) {
          world.cur_idx[DIM_X]--;
          new_map();
        }
        break;
      case 'q':
        // Quit
        break;
      case 'f':
        // Fly to coordinates
        scanf(" %d %d", &x, &y);
        if (x >= -(WORLD_SIZE / 2) && x <= WORLD_SIZE / 2 &&
            y >= -(WORLD_SIZE / 2) && y <= WORLD_SIZE / 2) {
          world.cur_idx[DIM_X] = x + (WORLD_SIZE / 2);
          world.cur_idx[DIM_Y] = y + (WORLD_SIZE / 2);
          new_map();
        }
        break;
      case '?':
      case 'h':
        // Help
        printf("Move with 'e'ast, 'w'est, 'n'orth, 's'outh or 'f'ly x y.\n"
               "Quit with 'q'.  '?' and 'h' print this help message.\n");
        break;
      default:
        fprintf(stderr, "%c: Invalid input.  Enter '?' for help.\n", c);
        break;
    }

    // Pause for a short time
    usleep(250000);

  } while (c != 'q');

  // Cleanup
  delete_world();

  printf("But how are you going to be the very best if you quit?\n");

  return 0;
}
