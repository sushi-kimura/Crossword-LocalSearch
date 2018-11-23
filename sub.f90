!!!!!! sub.f90 !!!!!!
! This program has some of the subroutines.
! subroutine list is written bellow:
!
! addab(a, b)
! addab2c(a, b, c)
! calc(n, Dict, puzzle, cover, mode)
!

!!! addab !!!
subroutine addab(a, b)
  integer a, b
  a = a + b
  b = a + b
  return
end subroutine addab

!!! addab2c !!!
subroutine addab2c(a, b, c)
  integer a, b, c
  c = a + b
  return
end subroutine addab2c
