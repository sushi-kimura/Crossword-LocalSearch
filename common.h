/******************** common.h ********************/
/***** valiables *****/
Dictionary Dict;
int n;          // 盤面の次数
Tuple *T;       // すべての可能なタプル集合
int *Sol;       // 解であるタプルの添字集合
int t_size;     // T に属するタプルの個数

/***** subroutines *****/
int add(int sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
int drop(int sol_size, int **puzzle, int **enable, int *score, int **cover, int t);
int calc(int **puzzle, int **cover, int mode);
int check_add(int sol_size, int t, int **puzzle, int **enable);
int check_drop(int sol_size, int **cover, int t);
void kick(int *sol_size, int **puzzle, int **enable, int *score, int **cover);
void shuffle(int arr[], int size);
void display(int **puzzle, int *score, int sol_size);
int move(int ****InvT, int **puzzle, int *score, int sol_size, int **enable, int **cover);
int check_move(int **puzzle, int **enable, int **cover, int direction);
int calc_profit();
void breakConnection(int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
