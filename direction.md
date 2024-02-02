기한
1주일

Goal
이미지를 M * N으로 자른다
M * N 으로 조각난 이미지를 하나로 합친다
사용법과 본인의 해결방안을 정리하여 Readme를 작성한다

요구사항
Junior 지원자의 경우 2x2, 3x3을 구현해 주시면 됩니다 
Senior 지원자의 경우 M * N을 구현해 주시면 됩니다
필수 : (M <= 4, N <= 4)
옵션 : (M > 5,  N > 5)

위 목표에 적용되어야 하는 추가적인 요구사항 및 제약사항은 아래를 참고해주셨으면 합니다

Goal 1. 이미지 자르기 단계 요구사항
각 이미지 조각들을 각각 0.5 확률로 mirror, flip, rotate시켜 저장한다
rotate는 시계방향으로 90도 회전을 의미한다
각 sub image들은 세 가지 augmentation이 각각 적용될 수 있다
저장된 이미지의 이름은 원본 파일의 어느 조각인지, 어떤 augmentation이 사용되었는지를 추측할 수 없도록 10자리의 난수로 설정한다
원본 이미지 크기와 M, N이 나누어 떨어지지 않을 경우에는 원본 이미지에서 가로 및 세로 크기를 잘라내어 나누어 떨어지도록 조정한 뒤 사용한다
ex) Image : 100 x 100, M : 3, N : 3 일 때 Image size를 99x99가 되도록 가로, 세로를 한 줄씩 제거한다
작성한 스크립트는 아래와 같은 입력을 터미널 혹은 config file을 통해 입력받도록 제작한다.
{input_image_name} {M(row_num)} {N(col_num)} {save_dir}
ex) cut.py test.jpg 3 4 test_01
이미 save_dir이 존재한다면 내용물을 비운뒤 새로운 output을 저장하도록 구현해 주시기 바랍니다

Goal 2. 이미지 병합 단계 요구사항
이미지 자르기를 통해 생성된 sub image를 사용하여 각 조각들을 mirror, flip, rotate하여 하나의 이미지로 복원하는 스크립트를 제작해 주시면 됩니다
rotate가 이루어질 수 있기 때문에 입력 이미지의 크기가 800x600일 경우800x600, 600x800의 output scale이 나올 수 있습니다
병합된 이미지는 정상적인 경우 원본 이미지와 동일하거나, 좌우반전, 상하반전, 회전된 이미지 일 수 있습니다
이미지 병합 스크립트는 반드시 성공하는 경우는 발생하지 않습니다
이비지 병합 스크립트에서는 원본 이미지 정보를 사용할 수 없습니다
조각난 이미지의 정보만을 사용하여 재구성 하시면 됩니다
이미지 병합 스크립트는 아래와 같은 입력을 터미널 혹은 config file을 통해 입력받도록 제작한다.
{input_dir} {M} {N} {result_img_name}
ex) merge.py 3 4 result.jpg





Hint
이미지 병합을 처리할 때에 생성된 sub image들의 경계면(Edge) 정보를 활용하여 접근하시면 됩니다
이미지 병합 스크립트가 효율적으로 동작하기 위해서는 모든 edge를 mirror, flip, rotate시켜 비교하는 부분을 잘 구성하셔야 합니다
edge를 비교할 때 어떤 edge들이 자연스럽게 병합이 될 수 있는 edge인지를 측정하는 방법을 고민해보셔야 할 것 같습니다

제약사항
해당 문제를 어떠한 경로를 통해서 정보를 취득하시고 답을 제출하셔도 상관 없습니다.
인터넷 검색을 통해서 답을 찾고 제출하셔도 상관 없습니다
어떠한 Library를 사용하셔도 상관 없습니다
제출하신 Code를 기술면접때 설명해주셔야 합니다
Git repo는 private로 제출해 주시기 바랍니다
junhu.park@d-meta.ai dudco0040@d-meta.ai leeys@d-meta.ai 에게 접근 권한을 부여하여 제출하시면 됩니다

제출해야 하는 목록
이미지 자르기 스크립트
이미지 병합 스크립트
Readme
test image
requirements.txt(필요시)
이미지 자르기 및 병합 스크립트를 한번에 수행할 수 있는 sh file(선택사항)

언제, 어떤 종류의 질문이던 아래 주소로 문의해 주시기 바랍니다
junhu.park@d-meta.ai dudco0040@d-meta.ai leeys@d-meta.ai