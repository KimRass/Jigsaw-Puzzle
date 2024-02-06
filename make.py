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
    def randomly_transform(img):
        if random.random() < 0.5:
            img = vflip(img)
        if random.random() < 0.5:
            img = hflip(img)
        if random.random() < 0.5:
            img = rotate(img)
        return img

    def make(self, img, M, N):
        h, w, _ = img.shape
        sub_h = h // M
        sub_w = w // N
        patches = list()
        for row in range(M):
            for col in range(N):
                patch = img[
                    row * sub_h: (row + 1) * sub_h,
                    col * sub_w: (col + 1) * sub_w,
                    :,
                ]
                patch = self.randomly_transform(patch)
                patches.append(patch)
        return patches

    @staticmethod
    def empty_dir(trg_dir):
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
    def get_rand_num():
        return random.randint(10 ** 9, (10 ** 10) - 1)

    def save(self, img_path, M, N, save_dir):
        self.empty_dir(save_dir)

        img = load_image(img_path)
        patches = self.make(img, M=M, N=N)
        for patch in patches:
            rand_num = self.get_rand_num()
            save_image(patch, save_path=Path(save_dir)/f"{rand_num}.png")
        print(f"Completed splitting the image into {M} x {N} patches!")


def main():
    args = get_args()

    model = JigsawPuzzleMaker()
    model.save(
        img_path=args.IMG_PATH, M=args.M, N=args.N, save_dir=args.SAVE_DIR,
    )


if __name__ == "__main__":
    main()
