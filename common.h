/******************** common.h ********************/

/***** subroutines *****/
int add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
int drop(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int **puzzle, int **enable, int *score, int **cover, int t);
int calc(int n, Dictionary *Dict, int **puzzle, int **cover, int mode);
int check_add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable);
int check_drop(int n, Dictionary *Dict, Tuple *T, int sol_size, int **cover, int t);
void kick(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int **puzzle, int **enable, int *score, int **cover);
void shuffle(int arr[], int size);
void display(Dictionary *Dict, int **puzzle, int *score, int sol_size, int n);
int move(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int ****InvT, int **puzzle, int *score, int sol_size, int **enable, int **cover);
int check_move(int n, int **puzzle, int **enable, int **cover, int direction);
int calc_profit(Dictionary *Dict, Tuple *T, int t_size, int *Sol);
void breakConnection(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
