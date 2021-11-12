import os
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection
this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TerminalDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(TerminalDetection, self).__init__(args, objName, scriptName)

    @try_except()
    def post_process(self, name, detectBoxes):
        def isInside(box0, box1):
            xmin0, ymin0, xmax0, ymax0 = box0
            xmin1, ymin1, xmax1, ymax1 = box1
            if xmin1 > xmin0 and ymin1 > ymin0 and xmax1 < xmax0 and ymax1 < ymax0:
                return True
            else:
                return False

        terminals = [obj for obj in detectBoxes if obj[0] == "terminal"]
        others = [obj for obj in detectBoxes if obj not in terminals]

        results = []
        for i, terminal in enumerate(terminals):
            terminal_box = terminal[2:-1]
            inside_list = []

            for obj in others:
                obj_box = obj[2:-1]
                if isInside(terminal_box, obj_box):
                    inside_list.append(obj[0])

            if "remind" in inside_list:
                results.append(
                    {"name": name, 'objName': self.objName, 'label': "terminal", 'bbox': terminal_box, "cls": "working"})
            else:
                if "light_screen" in inside_list or "dark_screen" in inside_list:
                    results.append({"name": name, 'objName': self.objName, 'label': "terminal", 'bbox': terminal_box, "cls": "working"})
                else:
                    results.append({"name": name, 'objName': self.objName, 'label': "terminal", 'bbox': terminal_box, "cls": "shutdown"})

        return results

