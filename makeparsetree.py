def set_fraction(compounds):
    compound = []
    i = 0
    while i < len(compounds):
        now = compounds[i]
        if now['did'] == False and now['symbol'] == '-':
            nowX = (now['xmin'] + now['xmax']) / 2
            nowY = (now['ymin'] + now['ymax']) / 2
            j = i
            while True:
                if j == 0: break
                prev = compounds[j - 1]
                if prev['did']: break
                prevX = (prev['xmin'] + prev['xmax']) / 2
                prevY = (prev['ymin'] + prev['ymax']) / 2
                if now['xmin'] <= prevX and now['xmax'] >= prevX:
                    j -= 1
                else:
                    break
            up = []
            up_tmp = []
            down = []
            down_tmp = []
            while j < len(compounds):
                if j != i:
                    check = compounds[j]
                    checkX = (check['xmin'] + check['xmax']) / 2
                    checkY = (check['ymin'] + check['ymax']) / 2
                    if now['xmin'] <= checkX <= now['xmax']:
                        if now['ymax'] < checkY and now['ymax'] < checkY:
                            down.append(check)
                            down_tmp.append(check.copy())
                        elif now['ymax'] > checkY and now['ymin'] > checkY:
                            up.append(check)
                            up_tmp.append(check.copy())
                        else:
                            break

                j += 1
            if len(up) != 0 and len(down) != 0:
                for d in down:
                    d['did'] = True
                for u in up:
                    u['did'] = True
                now['did'] = True
                up = set_fraction(up_tmp)
                down = set_fraction(down_tmp)
                compound.append({
                    'class': 'fraction',
                    'symbol': '-',
                    'did': False,
                    'xmin': now['xmin'],
                    'xmax': now['xmax'],
                    'ymin': now['ymin'],
                    'ymax': now['ymax'],
                    'up': up,
                    'down': down,
                    'inner': [],
                })
        i += 1
    for i in range(0, len(compounds)):
        now = compounds[i]
        if not now['did']:
            compound.append({
                'class': 'normal',
                'symbol': now['symbol'],
                'did': False,
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'up': [],
                'down': [],
                'inner': [],
            })
        compound = sorted(compound, key=lambda k: (k['xmin'], k['ymin']))
    return compound


def set_square_root(bonding_boxes):
    compound = []
    i = 0
    while i < len(bonding_boxes):
        now = bonding_boxes[i]
        if now['did']:
            i += 1
            continue
        elif now['class'] == 'fraction':
            compound.append({
                'class': 'fraction',
                'symbol': '-',
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'did': False,
                'up': set_square_root(now['up']),
                'down': set_square_root(now['down']),
                'inner': [],
            })
            i += 1
            continue
        # 마지막 문자 && compound에 속해 있지 않는 경우 compound에 넣고 종료
        if i == len(bonding_boxes) - 1:
            now['did'] = True
            compound.append({
                'class': 'normal',
                'did': False,
                'symbol': now['symbol'],
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'up': [],
                'down': [],
                'inner': [],
            })
            i += 1
            continue
        # square root를 재귀적으로 짜야한다.
        inner = []
        while i < len(bonding_boxes) - 1:
            later = bonding_boxes[i + 1]
            if later['did']:
                continue
            laterX = (later['xmin'] + later['xmax']) / 2
            laterY = (later['ymin'] + later['ymax']) / 2
            if now['xmin'] <= laterX <= now['xmax'] and now['ymin'] <= laterY <= now['ymax']:
                inner.append(later.copy())
                later['did'] = True
            else:
                break
            i += 1
            # if now['xmin'] < later['xmin'] and now['ymin'] < later['ymin'] and \
            #         now['xmax'] > later['xmax'] and now['ymax'] > later['ymax']:
            #     inner.append(later.copy())
            #     later['did'] = True
            # else:
            #     break
            # i += 1

        if len(inner) == 0:
            now['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': now['symbol'],
                'did': False,
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'up': [],
                'down': [],
                'inner': []
            })
        else:
            compound.append({
                'class': 'square_root',
                'symbol': '√',
                'did': False,
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
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
        now = compounds[i]
        if now['did']:
            i += 1
            continue
        elif now['class'] == 'fraction':
            compound.append({
                'class': 'fraction',
                'symbol': '-',
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'did': False,
                'up': set_script(now['up']),
                'down': set_script(now['down']),
                'inner': [],
            })
            i += 1
            continue
        elif now['class'] == 'square_root':
            compound.append({
                'class': 'square_root',
                'symbol': '√',
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'did': False,
                'up': [],
                'down': [],
                'inner': set_script(now['inner']),
            })
            i += 1
            continue
        # 마지막 문자 && compound에 속해 있지 않는 경우 compound에 넣고 종료
        # if i == len(compounds) - 1:
        #     now['did'] = True
        #     compound.append({
        #         'class': 'normal',
        #         'did': False,
        #         'symbol': now['symbol'],
        #         'xmin': now['xmin'],
        #         'xmax': now['xmax'],
        #         'ymin': now['ymin'],
        #         'ymax': now['ymax'],
        #         'up': [],
        #         'down': [],
        #         'inner': [],
        #     })
        #     i += 1
        #     continue

            # if fraction symbol is detected, divide up, down while changing 'did' flags
        up = []
        while i < len(compounds) - 1:
            later = compounds[i + 1]
            if later['did']:
                continue
            k = 0.5
            if now['ymin'] >= later['ymin'] and now['ymin'] + k * (now['ymax'] - now['ymin']) >= later['ymax']:
                up.append(later.copy())
                later['did'] = True
            else:
                break
            i += 1

        if len(up) == 0:
            now['did'] = True
            compound.append({
                'class': 'normal',
                'symbol': now['symbol'],
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'up': [],
                'down': [],
                'inner': [],
                'did': False,
            })
        else:
            now['did'] = True
            up = set_script(up)
            compound.append({
                'class': 'script',
                'symbol': now['symbol'],
                'did': False,
                'xmin': now['xmin'],
                'xmax': now['xmax'],
                'ymin': now['ymin'],
                'ymax': now['ymax'],
                'up': up,
                'down': [],
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
                ret += '!frac{'
                ret += print_compounds(up)
                ret += "}{"
                ret += print_compounds(down)
                ret += "}"
            elif classification == 'square_root':
                ret += '!sqrt{'
                ret += print_compounds(inner)
                ret += "}"
            elif classification == 'script':
                ret += symbol
                if len(up) != 0:
                    ret += '^{'
                    ret += print_compounds(up)
                    ret += '}'
            else:
                ret += symbol
    return ret
