/******************** common.h ********************/
int n;
Dictionary Dict;
Tuple *T;       // すべての可能なタプル集合
/***** subroutines *****/
int add(int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
int drop(int t_size, int *Sol, int sol_size, int **puzzle, int **enable, int *score, int **cover, int t);
int calc(int **puzzle, int **cover, int mode);
int check_add(int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable);
int check_drop(int sol_size, int **cover, int t);
void kick(int t_size, int *Sol, int *sol_size, int **puzzle, int **enable, int *score, int **cover);
void shuffle(int arr[], int size);
void display(int **puzzle, int *score, int sol_size, int n);
int move(int t_size, int *Sol, int ****InvT, int **puzzle, int *score, int sol_size, int **enable, int **cover);
int check_move(int **puzzle, int **enable, int **cover, int direction);
int calc_profit(int t_size, int *Sol);
void breakConnection(int t_size, int *Sol, int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
