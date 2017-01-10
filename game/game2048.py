from Tkinter import Tk, Label, Frame, BOTH
from tkFont import Font
from numpy import where, random, array, zeros, append, sum, concatenate, copy, ndenumerate, isnan, rot90, nan, int, float
DOWN, RIGHT, UP, LEFT = range(4)

color_map = {2: ('#776e65', '#eee4da'), 4: ('#776e65', '#ede0c8'), 8: ('#f9f6f2', '#f2b179'), 16: ('#f9f6f2', '#f2b179'),
             32: ('#f9f6f2', '#f67c5f'), 64: ('#f9f6f2', '#f65e3b'), 128:('#f9f6f2', '#edcf72'), 256: ('#f9f6f2', '#edcc61'),
             512: ('#f9f6f2', '#edc850'), 1024: ('#f9f6f2', '#edc53f'), 2048: ('#f9f6f2', '#edc22e'), 'base': '#ccc0b3'}
color_map.update(dict.fromkeys([2**x for x in range(12, 18)], ('#f9f6f2', '#3c3a32')))

root, window_size = Tk(), 360
root.title('Move tiles to get 2048! Score: 0')
root.geometry('{0}x{0}+111+111'.format(window_size))
root.config(background='#bbada0')

class Game2048:
    def __init__(self):
        self._grid, self._score = zeros(16) + nan, 0
        self._grid[random.choice(16, 2, replace=False)] = random.choice([2]*9+[4], 2, replace=False) # init with 2 tiles
        self._grid = self._grid.reshape((4, 4))  # create 4x4 grid

        grid, labels = self.get_grid(), []        
        for (i, j), value in ndenumerate(grid):
            frame = Frame(root, width=window_size/4-2, height=window_size/4-2)
            font = Font(family='Helvetica', weight='bold', size=window_size/15)
            frame.pack_propagate(0)
            frame.place(x=j*window_size/4+1, y=i*window_size/4+1)
            (text, color) = ('', color_map['base']) if isnan(value) else ('{}'.format(int(value)), color_map[value][0])
            label = Label(frame, text=text, font=font, fg=color, bg=color_map['base'] if isnan(value) else color_map[value][1])
            label.pack(fill=BOTH, expand=True)
            labels.append(label)
        self.labels = labels
        #root.mainloop()
        root.update()

    @staticmethod
    def _merge_down(grid):
        merge = concatenate((grid, [zeros(4) + nan])) - concatenate(([zeros(4) + nan], grid))  # find the mergable tiles
        merge[2][merge[3]==0], merge[1][merge[2]==0] = nan, nan     # remove redundant 0 by 3 same tiles
        score = sum(grid[merge[:4] == 0])
        grid[merge[:4] == 0], grid[merge[1:] == 0] = grid[merge[:4] == 0] * 2, nan # fill the merged  with new number
        return score

    def _create_tiles(self):
        avail = isnan(self._grid)
        if avail[avail==True].size > 0:
            new_tiles = append(random.choice([20]*9+[40]), zeros(avail[avail==True].size - 1) + nan)
            random.shuffle(new_tiles)
            self._grid[avail] = new_tiles
    #[R U 0 0]
    def step(self, direct):
        root.update()
        direction = where(direct == 1)[0][0]
        self._grid[self._grid%10==0] /= 10
        merge_v, merge_h, grid_copy = copy(self._grid), copy(rot90(self._grid)), copy(self._grid)
        map(Game2048._merge_down, [merge_v, merge_h])       # try to merge tiles along two directions
        if merge_v[isnan(merge_v)].size is 0 and merge_h[isnan(merge_h)].size is 0:         # Check if game is over
            grid, new_tiles, score = self.get_grid(), self.get_new_tiles(), int(self.get_score())
            max_tile = int(grid[~isnan(grid)].max())
            [self.labels[i].config(text='' if i < 4 or i > 11 else 'GAMEOVER'[i-4], bg=color_map['base']) for i in xrange(16)]
            root.title('Game Over! Tile acheived: {}, Score: {}'.format(max_tile, score))
            return self.get_grid(),-1,True
        self._grid = rot90(self._grid, RIGHT - direction)
        self._grid = array([concatenate((x[isnan(x)], x[~isnan(x)])) for x in self._grid])  # move tiles
        self._grid = rot90(self._grid, -1)
        t_score = Game2048._merge_down(self._grid)
        self._score += t_score                                     # merge tiles
        self._grid = rot90(self._grid, 1)
        self._grid = array([concatenate((x[isnan(x)], x[~isnan(x)])) for x in self._grid])  # move tiles
        self._grid = rot90(self._grid, direction - RIGHT)
        if not ((self._grid == grid_copy) | (isnan(self._grid) & isnan(grid_copy))).all():
            self._create_tiles()
        grid, new_tiles, score = self.get_grid(), self.get_new_tiles(), int(self.get_score())
        max_tile = int(grid[~isnan(grid)].max())
        root.title('Move tiles to get {}! Score: {}'.format(2048 if max_tile < 2048 else max_tile * 2, score))
        for (i, j), value in ndenumerate(grid):
            text = '{}'.format('' if isnan(grid[i][j]) else int(grid[i][j]))
            font_color = color_map[32][1] if new_tiles[i][j] else color_map['base'] if isnan(value) else color_map[value][0]
            self.labels[4*i+j].config(text=text, fg=font_color, bg=color_map['base'] if isnan(value) else color_map[value][1])                                
        return self.get_grid(),t_score,False

    def get_grid(self):
        grid = copy(self._grid)
        grid[grid%10==0] /= 10
        return grid

    def get_new_tiles(self):
        grid = zeros((4, 4), int)
        grid[self._grid%10==0] = 1
        return grid

    def get_score(self):
        return self._score
