# -*- coding: utf-8  -*-
# -*- author: jokker -*-

"""歌词解析，格式很简单，可以写一个脚本进行解析"""

from ..utils.StrUtil import StrUtil


class LyricsUtil(object):

    @staticmethod
    def get_lyrics_info(lrc_path, code_mode='utf-8'):
        """获取歌词信息"""
        lyrics_info = {'ti': None, 'ar': None, 'al': None, 'by': None, 'content': []}  # 保存歌词信息的数据结构

        lrc_file = open(lrc_path, 'r', encoding=code_mode)  # utf-8 | GBK

        for each_line in lrc_file:
            if StrUtil.match(each_line.strip(), r'\[\d{2}:\d{2}\.\d{2}.*\].*'):
                time_str, song_str = each_line.strip()[1:9], each_line.strip()[10:]
                lyrics_info['content'].append((time_str, song_str))

            elif StrUtil.match(each_line.strip(), r'\[ti:.*\].*'):  #
                lyrics_info['ti'] = StrUtil.find_all(each_line.strip(), r'\[ti:(.*)\].*')[0]
            elif StrUtil.match(each_line.strip(), r'\[ar:.*\].*'):  #
                lyrics_info['ar'] = StrUtil.find_all(each_line.strip(), r'\[ar:(.*)\].*')[0]
            elif StrUtil.match(each_line.strip(), r'\[al:.*\].*'):  #
                lyrics_info['al'] = StrUtil.find_all(each_line.strip(), r'\[al:(.*)\].*')[0]
            elif StrUtil.match(each_line.strip(), r'\[by:.*\].*'):  #
                lyrics_info['by'] = StrUtil.find_all(each_line.strip(), r'\[by:(.*)\].*')[0]

        lrc_file.close()

        return lyrics_info


if __name__ == "__main__":

    # lrcPath = r'D:\Code\Util_Util\LyricsUtil\data\QHC.lrc'
    lrcPath = r'D:\Code\Util_Util\Z_Example\LyricsUtil\data\test_002.lrc'

    a = LyricsUtil.get_lyrics_info(lrcPath)

    for each in a.items():
        print(each)

