from pathlib import Path
import argparse
import random

from utils import load_image, vflip, hflip, rotate, save_image


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


class JigsawPuzzleMaker(object):
    @staticmethod
    def _randomly_transform(img):
        """
        `img`에 상하반전, 좌우반전, 시계방향의 90° 회전을 각각 50%의 확률로 적용합니다.
        """
        if random.random() < 0.5:
            img = vflip(img)
        if random.random() < 0.5:
            img = hflip(img)
        if random.random() < 0.5:
            img = rotate(img)
        return img

    @staticmethod
    def _empty_dir(trg_dir):
        """
        `input_dir`을 비웁니다.
        """
        try:
            path = Path(trg_dir)
            for item in path.glob('*'):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            
            print(f"Emptied the directory'{trg_dir}'!")
        except Exception as e:
            print(f"Error occured while trying to empty '{trg_dir}';\n{e}")

    @staticmethod
    def _get_rand_num():
        """
        무작위로 10의 자리의 자연수를 생성합니다.
        """
        return random.randint(10 ** 9, (10 ** 10) - 1)

    def make(self, img, M, N):
        """
        `img`에 대해 `M` × `N`의 Jigsaw puzzle을 만듭니다.
        """
        h, w, _ = img.shape
        sub_h = h // M
        sub_w = w // N
        pieces = list()
        for row in range(M):
            for col in range(N):
                piece = img[
                    row * sub_h: (row + 1) * sub_h,
                    col * sub_w: (col + 1) * sub_w,
                    :,
                ]
                piece = self._randomly_transform(piece)
                pieces.append(piece)
        return pieces

    def save(self, img_path, M, N, save_dir):
        """
        `M` × `N`의 Jigsaw puzzle을 만들고 `save_dir`에 저장합니다.
        """
        self._empty_dir(save_dir)

        img = load_image(img_path)
        pieces = self.make(img, M=M, N=N)
        for piece in pieces:
            rand_num = self._get_rand_num()
            save_image(piece, save_path=Path(save_dir)/f"{rand_num}.png")
        print(f"Completed splitting the image into {M} x {N} pieces!")


def main():
    args = get_args()

    model = JigsawPuzzleMaker()
    model.save(
        img_path=args.IMG_PATH, M=args.M, N=args.N, save_dir=args.SAVE_DIR,
    )


if __name__ == "__main__":
    main()
