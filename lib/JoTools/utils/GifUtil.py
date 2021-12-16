# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import imageio
from PIL import Image


class GifUtil(object):

    def __init__(self, gif_path):
        self.gif_mat = imageio.imread(gif_path)

    @staticmethod
    def img_list_to_gif(img_list, save_path, time_gap=0.5):
        """ 将 img 地址列表转换为 gif """
        frames = []
        for image_name in img_list:
            frames.append(imageio.imread(image_name))
        imageio.mimsave(save_path, frames, 'GIF', duration=time_gap)

    @staticmethod
    def get_array():
        """获取每一帧图片的数组，返回四维数组"""
        pass

    @staticmethod
    def __analyse_image(path):
        """Pre-process pass over the image to determine the mode (full or additive).
        Necessary as assessing single frames isn't reliable. Need to know the mode
        before processing all frames."""
        im = Image.open(path)
        results = {
            'size': im.size,
            'mode': 'full',
        }
        try:
            while True:
                if im.tile:
                    tile = im.tile[0]
                    update_region = tile[1]
                    update_region_dimensions = update_region[2:]
                    if update_region_dimensions != im.size:  # 查看是否存在不同尺寸的图片
                        results['mode'] = 'partial'
                        break
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        return results

    @staticmethod
    def split(path, save_path):
        """ Iterate the GIF, extracting each frame """
        mode = GifUtil.__analyse_image(path)['mode']
        im = Image.open(path)
        i = 0
        p = im.getpalette()
        last_frame = im.convert('RGBA')

        try:
            while True:
                print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))
                '''
                If the GIF uses local colour tables, each frame will have its own palette.
                If not, we need to apply the global palette to the new frame.
                '''
                if not im.getpalette():
                    im.putpalette(p)

                new_frame = Image.new('RGBA', im.size)

                '''
                Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
                If so, we need to construct the new frame by pasting it on top of the preceding frames.
                '''
                if mode == 'partial':
                    new_frame.paste(last_frame)

                new_frame.paste(im, (0, 0), im.convert('RGBA'))
                new_frame.save(r'{0}\{1}.png'.format(save_path, i))

                i += 1
                last_frame = new_frame
                im.seek(im.tell() + 1)
        except EOFError:
            pass

    @staticmethod
    def gif_upend(gif_path, save_path):
        """将gif进行倒放"""
        pass


if __name__ == "__main__":





    pass