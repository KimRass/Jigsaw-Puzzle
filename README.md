# 1. How to Use
- Python 3.9에서 이 저장소의 코드가 정상 작동함을 확인했습니다. 설치가 어려운 라이브러리는 사용되지 않습니다.
- 조각들이 모두 정사각형일 때에도 정상적으로 작동합니다.
## 1) Making Jigsaw Puzzle
```bash
# e.g.,
python3 make.py\
    --img_path="./test/test_image.jpg"\
    --M=8\ # Number of row splits
    --N=8\ # Number of column splits
    --save_dir="./test/made"\ # Directory to save Jigsaw puzzle pieces
```
- `M`, `N`은 자연수로서 크기에 제한은 없습니다.
## 2) Solving Jigsaw Puzzle
```bash
python3 solve.py\
    --input_dir="./test/made"\
    --M=8\
    --N=8\
    --save_path="./test/solved.png"\
```
- `M`, `N`은 자연수로서 크기에 제한은 없습니다.

# 2. Implementation Details
## 1) Making Jigsaw Puzzle
- 상하반전, 좌우반전, 시계방향의 90° 회전을 NumPy array의 연산만으로 구현했습니다.
## 2) Solving Jigsaw Puzzle
- Rule-based 알고리즘을 설계했습니다. 딥 러닝을 이용할 경우 다양한 조각의 수에 대해 모두 학습시켜야 하고 또한 각 조각의 이미지 변환을 모두 예측하도록 모델을 훈련시켜야 하는 등 어려움에 존재한다고 판단했습니다.
- 서로 다른 두 조각을 맞닿았을 때 인접한 각 조각의 변 사이의 L2 distance를 계산하고 이 값이 작다면 그 두 조각이 맞닿는 것이 맞다는 것이 기본 아이디어입니다.
- 서로 다른 두 조각을 선택할 수 있는 모든 경우의 수에 대해서, 이 두 조각을 가지고 만들 수 있는 변들간의 L2 distance에 따라 경우의 수를 오름차순 정렬합니다. 이 순서에 따라 조각을 적절한 위치에 배치해나갑니다. 

# 3. References
- [1] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
