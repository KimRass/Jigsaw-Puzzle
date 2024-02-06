# 1. How to Use
## 1) Making Jigsaw Puzzle
```bash
# e.g.,
python3 make.py\
    --img_path="./test/test_image.jpg"\
    --M=8\ # Number of row splits
    --N=8\ # Number of column splits
    --save_dir="./test/made"\ # Directory to save Jigsaw puzzle pieces
```
## 2) Solving Jigsaw Puzzle
```bash
python3 solve.py\
    --input_dir="./test/made"\
    --M=8\
    --N=8\
    --save_path="./test/solved.png"\
```

# 2. Implementation Details
## 1) Making Jigsaw Puzzle
- 상하반전, 좌우반전, 시계방향의 90° 회전을 NumPy array의 연산만으로 구현했습니다.
## 2) Solving Jigsaw Puzzle



### (1) Learning-based
- 사용자가 어떤 이미지를 넣을지 모르기 떄문에 학습에 사용된 데이터의 분포를 크게 벗어난 이미지를 넣는다면 잘 작동하지 않을 것.
- 데이터의 분포를 넓히기 위해서는 너무 많은 데이터가 필요함.

# 3. References
- [1] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
