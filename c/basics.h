
/******************** basics.h ********************/


/***** macros *****/
#define MAXCODE_EN     90
#define MINCODE_EN     65
#define MAXCODE_JA     12541
#define MINCODE_JA     12353
#define WORD_MAX       36
#define True   -1
#define False  -2
#define Undef  -3
#define BLACK_CELL 0
#define WHITE_CELL 1
#define NULL_WORD 0


/***** typedefs *****/
typedef int Boolean;


/***** structure *****/
typedef struct{
  int n;
  int **w;      // words with original letters
  int **x;      // extended words with reduced letters
  int *len;     // lengths of words
  int *p;       // profit of words
  int size;     // number of words
  int m;        // number of valid words (words longer than n are excluded)
  int en;       // whether English dictionary (1) or not (0)
  int *dist;    // distribution of letters
  int letters;  // number of possible letters (=size of dist)
  int *reduce;  // mapping from original letters to reduced ones
  int reds;     // number of reduced letters (including blacks)
  int *inv;     // inverse mapping
} Dictionary;

typedef struct {
	int div;    // 0:vertical 1:horizontal
	int k;
	int i;
	int j;
} Tuple;

/***** subroutines *****/
FILE *openFile(char *fname, char *mode);
int getDictSize(FILE *fp);
Boolean isEnglish(FILE *fp);
void initDict(Dictionary *D, int n, char *fname);
void readDict(Dictionary *D, char *fname, int type);
void readDictEn(Dictionary *D, FILE *fp);
void readDictEnWithWeight(Dictionary *D, FILE *fp);
void readDictJa(Dictionary *D, FILE *fp);
int getHeadOnes(char c);
char *decoder(int p);
int encoder(char x, char y);
int getConnectedComponents(int n, int **G);
void DFS(int n, int **G, int i, int j, int m);
