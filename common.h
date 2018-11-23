/******************** common.h ********************/
/***** Global valiables *****/
Dictionary Dict;
int n;          // 盤面の次数
Tuple *T;       // すべての可能なタプル集合
int *Sol;       // 解であるタプルの添字集合
int t_size;     // T に属するタプルの個数
int sol_size;   // 解 Sol に属するタプルの個数
int score;      //目的関数値
int **puzzle;
int **enable;
int **cover;

/***** subroutines *****/
int add(int t);
int drop(int t);
int calc(int mode);
int check_add(int t);
int check_drop(int t);
void kick();
void shuffle(int arr[], int size);
void display();
int move(int ****InvT);
int check_move(int direction);
int calc_profit();
void breakConnection(int t);
