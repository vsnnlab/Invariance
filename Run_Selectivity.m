   %       R   e   s   u   l   t       1   :       E   m   e   r   g   e   n   c   e       o   f       s   e   l   e   c   t   i   v   i   t   y       t   o       v   a   r   i   o   u   s       o   b   j   e   c   t   s       i   n       u   n   t   r   a   i   n   e   d       n   e   t   w   o   r   k   s   
       
   f   i   g   u   r   e   (   '   u   n   i   t   s   '   ,   '   n   o   r   m   a   l   i   z   e   d   '   ,   '   o   u   t   e   r   p   o   s   i   t   i   o   n   '   ,   [   0       0   .   2       1       0   .   8   ]   )   ;   
   s   g   t   i   t   l   e   (   '   R   e   s   u   l   t       1       :       E   m   e   r   g   e   n   c   e       o   f       s   e   l   e   c   t   i   v   i   t   y       t   o       v   a   r   i   o   u   s       o   b   j   e   c   t   s       i   n       u   n   t   r   a   i   n   e   d       n   e   t   w   o   r   k   s   '   )   
       
   %   %       L   o   w   -   l   e   v   e   l       f   e   a   t   u   r   e   -   c   o   n   t   r   o   l   l   e   d       s   t   i   m   u   l   u   s       (   F   i   g   u   r   e       1   A   ,       S   u   p   p   l   e       F   i   g   u   r   e       1   )   
   s   t   i   m   u   l   u   s   _   i   d   x       =       [   1   :   5   ,   1   1   +   1   :   1   1   +   5   ]   ;   
   f   o   r       i   i   =   1   :   1   0   
                   s   u   b   p   l   o   t   (   4   ,   1   1   ,   s   t   i   m   u   l   u   s   _   i   d   x   (   i   i   )   )   ;   
                   i   m   a   g   e   s   c   (   I   M   G   _   O   R   I   (   :   ,   :   ,   1   ,   n   u   m   I   M   G   *   (   i   i   -   1   )   +   1   )   )   ;   
                   c   o   l   o   r   m   a   p   (   '   g   r   a   y   '   )   ;   
                   a   x   i   s       s   q   u   a   r   e   ;   
                   a   x   i   s       o   f   f   ;   
                   t   i   t   l   e   (   S   T   R   _   L   A   B   E   L   (   i   i   )   ,   '   I   n   t   e   r   p   r   e   t   e   r   '   ,   '   n   o   n   e   '   )   ;   
   e   n   d   
   
   n   e   t   _   r   a   n   d       =       C   e   l   l   _   N   e   t   {   1   }   ;           
       
   %   %       O   b   j   e   c   t   -   s   e   l   e   c   t   i   v   e       r   e   s   p   o   n   s   e       (   F   i   g   u   r   e       1   C   ,       l   e   f   t   )   
   n   u   m   _   c   e   l   l       =       p   r   o   d   (   a   r   r   a   y   _   s   z   (   l   a   y   e   r   A   r   r   a   y   (   5   )   ,   :   )   )   ;   
   n   e   t   _   r   a   n   d       =       C   e   l   l   _   N   e   t   {   1   }   ;       I   d   x   _   T   a   r   g   e   t       =       C   e   l   l   _   I   d   x   {   1   ,   5   }   ;   
       
   a   c   t   _   r   a   n   d       =       a   c   t   i   v   a   t   i   o   n   s   (   n   e   t   _   r   a   n   d   ,   I   M   G   _   O   R   I   ,   l   a   y   e   r   s   S   e   t   {   l   a   y   e   r   A   r   r   a   y   (   5   )   }   )   ;   
   [   r   e   p   _   m   a   t   ,   r   e   p   _   m   a   t   _   3   D   ]       =       f   u   n   _   R   e   s   Z   s   c   o   r   e   (   a   c   t   _   r   a   n   d   ,   n   u   m   _   c   e   l   l   ,   I   d   x   _   T   a   r   g   e   t   ,   n   u   m   C   L   S   ,   n   u   m   I   M   G   )   ;   
       
   a   x   2       =       s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   7   :   9   ,   1   1   +   7   :   1   1   +   9   ]   )   ;   
   l   o   a   d   (   '   C   o   l   o   r   b   a   r   _   T   s   a   o   .   m   a   t   '   )   ;   
   i   m   a   g   e   s   c   (   r   e   p   _   m   a   t   )   ;   
   c   a   x   i   s   (   [   -   3       3   ]   )   
   f   o   r       c   c       =       1   :   n   u   m   C   L   S   -   1   
   l   i   n   e   (   [   n   u   m   I   M   G   *   c   c       n   u   m   I   M   G   *   c   c   ]   ,       [   1       l   e   n   g   t   h   (   I   d   x   _   T   a   r   g   e   t   )   ]   ,   '   c   o   l   o   r   '   ,   '   k   '   ,   '   L   i   n   e   S   t   y   l   e   '   ,   '   -   -   '   )   
   e   n   d   
   x   t   i   c   k   s   (   [   n   u   m   I   M   G   /   2   :   n   u   m   I   M   G   :   n   u   m   I   M   G   *   (   n   u   m   C   L   S   +   0   .   5   )   ]   )   ;       x   t   i   c   k   l   a   b   e   l   s   (   S   T   R   _   L   A   B   E   L   )   ;   
   c       =       c   o   l   o   r   b   a   r   ;       c   o   l   o   r   m   a   p   (   a   x   2   ,   c   m   a   p   )   ;   
   c   .   L   a   b   e   l   .   S   t   r   i   n   g       =       '   R   e   s   p   o   n   s   e       (   z   -   s   c   o   r   e   d   )   '   ;   
   y   l   a   b   e   l   (   '   U   n   i   t       i   n   d   i   c   e   s   '   )   ;       t   i   t   l   e   (   '   R   e   s   p   o   n   s   e   s       o   f       o   b   j   e   c   t   -   s   e   l   e   c   t   i   v   e       u   n   i   t   s   '   )   ;       c   l   e   a   r   v   a   r   s       c   m   a   p   
   s   e   t   (   g   c   a   ,   '   T   i   c   k   L   a   b   e   l   I   n   t   e   r   p   r   e   t   e   r   '   ,   '   n   o   n   e   '   ,   '   T   i   c   k   D   i   r   '   ,   '   o   u   t   '   )   ;   
       
   %   %       O   b   j   e   c   t   -   s   e   l   e   c   t   i   v   i   t   y       i   n   d   e   x   
   r   e   p   _   s   h   u   f   _   m   a   t   _   3   D       =       r   e   s   h   a   p   e   (   r   e   p   _   m   a   t   _   3   D   (   r   a   n   d   p   e   r   m   (   n   u   m   e   l   (   r   e   p   _   m   a   t   _   3   D   )   )   )   ,   s   i   z   e   (   r   e   p   _   m   a   t   _   3   D   )   )   ;   
   o   s   i   _   m   a   t       =       f   u   n   _   O   S   I   (   r   e   p   _   m   a   t   _   3   D   )   ;   
   o   s   i   _   s   h   u   f   _   m   a   t       =       f   u   n   _   O   S   I   (   r   e   p   _   s   h   u   f   _   m   a   t   _   3   D   )   ;   
       
   s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   2   *   1   1   +   4   :   2   *   1   1   +   6   ,       3   *   1   1   +   4   :   3   *   1   1   +   6   ]   )   ;       h   o   l   d       o   n   ;   
   b   o   x   p   l   o   t   (   [   o   s   i   _   m   a   t   ,   o   s   i   _   s   h   u   f   _   m   a   t   ]   )   
   x   t   i   c   k   s   (   [   1   :   2   ]   )   ;       x   t   i   c   k   l   a   b   e   l   s   (   {   '   U   n   t   r   a   i   n   e   d   '   ,   '   R   e   s   p   o   n   s   e       s   h   u   f   f   l   e   d   '   }   )   ;       x   l   i   m   (   [   0   .   5       2   .   5   ]   )   ;   
   y   l   a   b   e   l   (   '   O   b   j   e   c   t   -   s   e   l   e   c   t   i   v   i   t   y       i   n   d   e   x   '   )   ;       t   i   t   l   e   (   '   S   i   n   g   l   e       n   e   u   r   o   n       t   u   n   i   n   g       '   )   ;   
   s   e   t   (   g   c   a   ,   '   T   i   c   k   L   a   b   e   l   I   n   t   e   r   p   r   e   t   e   r   '   ,   '   n   o   n   e   '   ,   '   T   i   c   k   D   i   r   '   ,   '   o   u   t   '   )   ;   
       
   %   %       S   i   n   g   l   e       u   n   i   t       t   u   n   i   n   g       c   u   r   v   e       (   F   i   g   u   r   e       1   C   ,       r   i   g   h   t   )   
   s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   1   0   ,   1   1   ,   1   1   +   1   0   ,   1   1   +   1   1   ]   )   ;       h   o   l   d       o   n   ;   
   m   a   x   _   o   s   i   _   i   d   x       =       f   i   n   d   (   o   s   i   _   m   a   t       =   =       m   a   x   (   o   s   i   _   m   a   t   )   )   ;   
   p   l   o   t   (   [   0   :   n   u   m   C   L   S   +   1   ]   ,   [   0       m   e   a   n   (   r   e   p   _   m   a   t   _   3   D   (   m   a   x   _   o   s   i   _   i   d   x   ,   :   ,   :   )   ,   3   )       0   ]   ,   '   c   o   l   o   r   '   ,   '   r   '   )   ;   
   x   l   i   m   (   [   0   .   5       n   u   m   C   L   S   +   0   .   5   ]   )   ;       x   t   i   c   k   s   (   [   1   :   n   u   m   C   L   S   ]   )   ;       x   t   i   c   k   l   a   b   e   l   s   (   S   T   R   _   L   A   B   E   L   )   ;   
   y   l   i   n   e   (   0   ,       '   -   -   '   )   ;   
   y   l   a   b   e   l   (   '   R   e   s   p   o   n   s   e       (   z   -   s   c   o   r   e   d   )   '   )   ;       t   i   t   l   e   (   [   '   T   u   n   i   n   g       c   u   r   v   e       o   f       o   b   j   e   c   t   -   s   e   l   e   c   t   i   v   e       u   n   i   t   s       (   #   '       n   u   m   2   s   t   r   (   m   a   x   _   o   s   i   _   i   d   x   )       '       u   n   i   t   )   '   ]   )   ;   
   s   e   t   (   g   c   a   ,   '   T   i   c   k   L   a   b   e   l   I   n   t   e   r   p   r   e   t   e   r   '   ,   '   n   o   n   e   '   ,   '   T   i   c   k   D   i   r   '   ,   '   o   u   t   '   )   ;   
       
   %   %       A   v   e   r   a   g   i   n   g       t   u   n   i   n   g       c   u   r   v   e       (   F   i   g   u   r   e       1   D   )   
   S   T   R   _   B   A   S   I   C       =           {   '   t   o   i   l   e   t   '   ,       '   b   e   d   '   ,   '   c   h   a   i   r   '   ,   '   d   e   s   k   '   ,   '   d   r   e   s   s   e   r   '   ,   '   m   o   n   i   t   o   r   '   ,       .   .   .   
                                   '   n   i   g   h   t   _   s   t   a   n   d   '   ,       '   s   o   f   a   '   ,       '   t   a   b   l   e   '   ,       '   s   c   r   a   m   b   l   e   d   '   }   ;   
                   
   s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   2   *   1   1   +   1   :   2   *   1   1   +   3   ,       3   *   1   1   +   1   :   3   *   1   1   +   3   ]   )   ;       h   o   l   d       o   n   ;   
   s   h   a   d   e   d   E   r   r   o   r   B   a   r   (   [   0   :   n   u   m   C   L   S   +   1   ]   ,   [   0       m   e   a   n   (   m   e   a   n   (   r   e   p   _   m   a   t   _   3   D   (   :   ,   :   ,   :   )   ,   3   )   ,   1   )       0   ]   ,   .   .   .   
                   [   0       s   t   d   (   m   e   a   n   (   r   e   p   _   m   a   t   _   3   D   (   :   ,   :   ,   :   )   ,   3   )   ,   0   ,   1   )       0   ]   )   
   x   l   i   m   (   [   0   .   5       n   u   m   C   L   S   +   0   .   5   ]   )   ;       x   t   i   c   k   s   (   [   1   :   n   u   m   C   L   S   ]   )   ;       x   t   i   c   k   l   a   b   e   l   s   (   S   T   R   _   L   A   B   E   L   )   ;   
   y   l   i   m   (   [   -   1       1   ]   )   ;   
   y   l   i   n   e   (   0   ,       '   -   -   '   )   ;   
   y   l   a   b   e   l   (   '   R   e   s   p   o   n   s   e       (   z   -   s   c   o   r   e   d   )   '   )   ;       t   i   t   l   e   (   '   A   v   e   r   a   g   e   d       t   u   n   i   n   g       c   u   r   v   e       o   f       o   b   j   e   c   t   -   s   e   l   e   c   t   i   v   e       u   n   i   t   s   '   )   ;   
   s   e   t   (   g   c   a   ,   '   T   i   c   k   L   a   b   e   l   I   n   t   e   r   p   r   e   t   e   r   '   ,   '   n   o   n   e   '   ,   '   T   i   c   k   D   i   r   '   ,   '   o   u   t   '   )   ;   
       
   %   %       C   l   u   s   t   e   r   i   n   g       i   n       l   a   t   e   n   t       s   p   a   c   e       (   t   -   S   N   E   )       (   F   i   g   u   r   e       1   F   ,       S   u   p   p   l   e       F   i   g   u   r   e       2   )   
       
   o   r   d   e   r   _   C   l   s   I   M   G       =       [   1       2       3       4       5       6       7       8       9   ]   ;                                                                                                                                                                   
       
   I   n   i   t   i   a   l   Y       =       1   e   -   4   *   r   a   n   d   n   (   s   i   z   e   (   I   M   G   _   O   R   I   ,   4   )   -   2   0   0   ,   2   )   ;       P   e   r   p   l   e   x   i   t   y       =       5   0   ;   
   l   a   b   e   l   s       =       [   ]   ;       f   o   r       c   c       =       1   :   n   u   m   C   L   S   -   1   ;       l   a   b   e   l   s       =       [   l   a   b   e   l   s   ;   c   c   .   *   o   n   e   s   (   n   u   m   I   M   G   ,   1   )   ]   ;   e   n   d   
   c   m   a   p       =       f   l   i   p   (   j   e   t   (   n   u   m   C   L   S   -   1   +   3   )   )   ;       c   m   a   p       =       c   m   a   p   (   r   o   u   n   d   (   l   i   n   s   p   a   c   e   (   1   ,   n   u   m   C   L   S   -   1   +   3   ,   n   u   m   C   L   S   -   1   +   1   )   )   ,   :   )   ;   
   c   m   a   p       =       c   m   a   p   (   2   :   e   n   d   ,   :   )   ;       c   m   a   p       =       c   m   a   p   (   o   r   d   e   r   _   C   l   s   I   M   G   ,   :   )   ;       s   z       =       1   0   ;   
       
   a   c   t   I   M   G       =       r   e   s   h   a   p   e   (   I   M   G   _   O   R   I   (   :   ,   :   ,   1   ,   1   :   1   8   0   0   )   ,   p   r   o   d   (   s   i   z   e   (   I   M   G   _   O   R   I   ,   1   :   2   )   )   ,   s   i   z   e   (   I   M   G   _   O   R   I   ,   4   )   -   2   0   0   )   ;       %       i   m   a   g   e   
   t   S   N   E   _   I   M   G       =       t   s   n   e   (   a   c   t   I   M   G   '   ,   '   I   n   i   t   i   a   l   Y   '   ,   I   n   i   t   i   a   l   Y   ,   '   P   e   r   p   l   e   x   i   t   y   '   ,   P   e   r   p   l   e   x   i   t   y   ,   '   S   t   a   n   d   a   r   d   i   z   e   '   ,   0   )   ;           
       
   l   a   y   e   r   _   i   d   x       =       5   ;                                                                                                                                                                                                                                                       %       c   o   n   v   5   
   n   u   m   _   c   e   l   l       =       p   r   o   d   (   a   r   r   a   y   _   s   z   (   l   a   y   e   r   _   i   d   x   ,   :   )   )   ;   
   a   c   t   _   r   a   n   d       =       a   c   t   i   v   a   t   i   o   n   s   (   n   e   t   _   r   a   n   d   ,   I   M   G   _   O   R   I   (   :   ,   :   ,   :   ,   1   :   1   8   0   0   )   ,   l   a   y   e   r   s   S   e   t   {   l   a   y   e   r   _   i   d   x   }   )   ;   
   a   c   t       =       r   e   s   h   a   p   e   (   a   c   t   _   r   a   n   d   ,   n   u   m   _   c   e   l   l   ,   s   i   z   e   (   I   M   G   _   O   R   I   ,   4   )   -   2   0   0   )   ;                                                                                           %       n   e   t   w   o   r   k       r   e   s   p   o   n   s   e   
   t   S   N   E   _   R   e   s   p       =       t   s   n   e   (   a   c   t   '   ,   '   I   n   i   t   i   a   l   Y   '   ,   I   n   i   t   i   a   l   Y   ,   '   P   e   r   p   l   e   x   i   t   y   '   ,   P   e   r   p   l   e   x   i   t   y   ,   '   S   t   a   n   d   a   r   d   i   z   e   '   ,   0   )   ;   
   
   s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   2   *   1   1   +   8   ,   2   *   1   1   +   9   ,   3   *   1   1   +   8   ,   3   *   1   1   +   9   ]   )   ;       h   o   l   d       o   n   
   f   o   r       i   i       =       n   u   m   C   L   S   -   1   :   -   1   :   1   
                   i   d   x       =       f   i   n   d   (   d   o   u   b   l   e   (   l   a   b   e   l   s   )       =   =       i   i   )   ;   
                   h       =       g   s   c   a   t   t   e   r   (   t   S   N   E   _   I   M   G   (   i   d   x   ,   1   )   ,   t   S   N   E   _   I   M   G   (   i   d   x   ,   2   )   ,   l   a   b   e   l   s   (   i   d   x   )   ,   c   m   a   p   (   i   i   ,   :   )   ,   '   .   '   ,   s   z   ,   '   o   f   f   '   )   ;   
   e   n   d   
   
   y   l   a   b   e   l   (   '   t   S   N   E       a   x   i   s       2   '   )   ;       x   l   i   m   (   [   -   5   0       5   0   ]   )   ;       y   l   i   m   (   [   -   5   0       5   0   ]   )   ;       t   i   t   l   e   (   '   R   a   w       i   m   a   g   e   s       (   t   -   S   N   E   )   '   )   ;   
       
   s   u   b   p   l   o   t   (   4   ,   1   1   ,   [   2   *   1   1   +   1   0   ,   2   *   1   1   +   1   1   ,   3   *   1   1   +   1   0   ,   3   *   1   1   +   1   1   ]   )   ;       h   o   l   d       o   n   
   f   o   r       i   i       =       n   u   m   C   L   S   -   1   :   -   1   :   1   
                   i   d   x       =       f   i   n   d   (   d   o   u   b   l   e   (   l   a   b   e   l   s   )       =   =       i   i   )   ;   
                   h       =       g   s   c   a   t   t   e   r   (   t   S   N   E   _   R   e   s   p   (   i   d   x   ,   1   )   ,   t   S   N   E   _   R   e   s   p   (   i   d   x   ,   2   )   ,   l   a   b   e   l   s   (   i   d   x   )   ,   c   m   a   p   (   i   i   ,   :   )   ,   '   .   '   ,   s   z   ,   '   o   f   f   '   )   ;   
   e   n   d   
   l   e   g   e   n   d   (   S   T   R   _   L   A   B   E   L   (   1   :   n   u   m   C   L   S   -   1   )   )   ;   
   x   l   a   b   e   l   (   '   t   S   N   E       a   x   i   s       1   '   )   ;       y   l   a   b   e   l   (   '   t   S   N   E       a   x   i   s       2   '   )   ;       x   l   i   m   (   [   -   5   0       5   0   ]   )   ;       y   l   i   m   (   [   -   5   0       5   0   ]   )   ;       t   i   t   l   e   (   '   C   o   n   v       r   e   s   p   o   n   s   e       i   n       u   n   t   r   a   i   n   e   d       n   e   t   w   o   r   k   s       (   t   -   S   N   E   )   '   )   ;   
