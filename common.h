/******************** common.h ********************/
/***** valiables *****/
Dictionary Dict;
int n;          // 盤面の次数
Tuple *T;       // すべての可能なタプル集合
int *Sol;       // 解であるタプルの添字集合
int t_size;     // T に属するタプルの個数
int **puzzle;
int **enable;
int **cover;
/***** subroutines *****/
int add(int sol_size, int t, int *score);
int drop(int sol_size, int *score, int t);
int calc(int mode);
int check_add(int sol_size, int t);
int check_drop(int sol_size, int t);
void kick(int *sol_size, int *score);
void shuffle(int arr[], int size);
void display(int *score, int sol_size);
int move(int ****InvT, int *score, int sol_size);
int check_move(int direction);
int calc_profit();
void breakConnection(int *sol_size, int t, int *score);
