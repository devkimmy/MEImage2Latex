def set_fraction(compounds):
    compound = []
    i = 0
    while i < len(compounds):
        box1 = compounds[i]
        if box1['did']:
            i += 1
            continue
        # 마지막 문자 && compound에 속해 있지 않는 경우 compound에 넣고 종료
        if i == len(compounds) - 1:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': box1['symbol'],
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': []
            })
            i += 1
            continue

        # if fraction symbol is detected, divide up, down while changing 'did' flags
        up = []
        down = []
        if box1['symbol'] == '-':
            # 현재위치부터 < 끝까지
            while i < len(compounds) - 1:
                box2 = compounds[i + 1]
                if box2['did']:
                    continue
                if box1['xmax'] >= box2['xmax'] and box1['xmin'] <= box2['xmin']:
                    if box1['ymin'] < box2['ymin']:
                        down.append(box2.copy())
                    else:
                        up.append(box2.copy())
                    box2['did'] = True
                else:
                    break
                i += 1
        if len(up) == 0 and len(down) == 0:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': box1['symbol'],
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': []
            })
        else:
            up = set_fraction(up)
            down = set_fraction(down)
            compound.append({
                'class': 'fraction',
                'symbol': '-',
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': up,
                'down': down,
                'inner': [],
            })
        i += 1
    return compound


def set_square_root(bonding_boxes):
    compound = []
    i = 0
    while i < len(bonding_boxes):
        box1 = bonding_boxes[i]
        if box1['did']:
            i += 1
            continue
        elif box1['class'] == 'fraction':
            compound.append({
                'class': 'fraction',
                'symbol': '-',
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'did': False,
                'up': set_square_root(box1['up']),
                'down': set_square_root(box1['down']),
                'inner': [],
            })
            i += 1
            continue
        # 마지막 문자 && compound에 속해 있지 않는 경우 compound에 넣고 종료
        if i == len(bonding_boxes) - 1:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'did': False,
                'symbol': box1['symbol'],
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': [],
            })
            i += 1
            continue
        # square root를 재귀적으로 짜야한다.
        inner = []
        while i < len(bonding_boxes) - 1:
            box2 = bonding_boxes[i + 1]
            if box2['did']:
                continue
            if box1['xmin'] < box2['xmin'] and box1['ymin'] < box2['ymin'] and \
                    box1['xmax'] > box2['xmax'] and box1['ymax'] > box2['ymax']:
                inner.append(box2.copy())
                box2['did'] = True
            else:
                break
            i += 1

        if len(inner) == 0:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': box1['symbol'],
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': []
            })
        else:
            compound.append({
                'class': 'square_root',
                'symbol': '√',
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': set_square_root(inner),
            })
        i += 1
    return compound


def set_script(compounds):
    compound = []
    i = 0
    while i < len(compounds):
        box1 = compounds[i]
        if box1['did']:
            i += 1
            continue
        elif box1['class'] == 'fraction':
            compound.append({
                'class': 'fraction',
                'symbol': '-',
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'did': False,
                'up': set_script(box1['up']),
                'down': set_script(box1['down']),
                'inner': [],
            })
            i += 1
            continue
        elif box1['class'] == 'square_root':
            compound.append({
                'class': 'square_root',
                'symbol': '√',
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'did': False,
                'up': [],
                'down': [],
                'inner': set_script(box1['inner']),
            })
            i += 1
            continue
        # 마지막 문자 && compound에 속해 있지 않는 경우 compound에 넣고 종료
        if i == len(compounds) - 1:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'did': False,
                'symbol': box1['symbol'],
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': [],
            })
            i += 1
            continue

            # if fraction symbol is detected, divide up, down while changing 'did' flags
        up = []
        down = []
        while i < len(compounds) - 1:
            box2 = compounds[i + 1]
            if box2['did']:
                continue
            k = 0.5
            if box1['ymin'] >= box2['ymin'] and box1['ymin'] + k * (box1['ymax'] - box1['ymin']) >= box2['ymax']:
                up.append(box2.copy())
                box2['did'] = True
            elif box1['ymax'] <= box2['ymax'] and box1['ymax'] - k * (box1['ymax'] - box1['ymin']) <= box2['ymin']:
                down.append(box2.copy())
                box2['did'] = True
            else:
                break
            i += 1

        if len(up) == 0 and len(down) == 0:
            box1['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': box1['symbol'],
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': [],
                'down': [],
                'inner': [],
                'did': False,
            })
        else:
            box1['did'] = True
            up = set_script(up)
            down = set_script(down)
            compound.append({
                'class': 'script',
                'symbol': box1['symbol'],
                'did': False,
                'xmin': box1['xmin'],
                'xmax': box1['xmax'],
                'ymin': box1['ymin'],
                'ymax': box1['ymax'],
                'up': up,
                'down': down,
                'inner': [],
            })
        i += 1
    return compound


def print_compounds(compounds):
    ret = ""
    if len(compounds) != 0:
        for i in range(len(compounds)):
            compound = compounds[i]
            up = compound['up']
            down = compound['down']
            inner = compound['inner']
            classification = compound['class']
            symbol = compound['symbol']
            if classification == 'fraction':
                ret += '\\frac{'
                ret += print_compounds(up)
                ret += "}{"
                ret += print_compounds(down)
                ret += "}"
            elif classification == 'square_root':
                ret += '\\sqrt{'
                ret += print_compounds(inner)
                ret += "}"
            elif classification == 'script':
                ret += symbol
                if len(down) != 0:
                    ret += '_{'
                    ret += print_compounds(down)
                    ret += '}'
                if len(up) != 0:
                    ret += '^{'
                    ret += print_compounds(up)
                    ret += '}'
            else:
                ret += symbol
    return ret
