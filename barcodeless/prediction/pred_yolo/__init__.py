from .darknet_cv import *
from .yolo_effi import *

def predict(IMAGE_PATH):
    rm_results()
    yolo(IMAGE_PATH)
    image_ls = yolo_result_list()
    count_dict = effi(image_ls)

    # print(count_dict)

    keys = list(count_dict.keys())
    # print(keys)
    cnt = list(count_dict.values())
    # print(cnt)

    price = {
        '라베스트' : 2000,
        '로투스' : 1000,
        '부라보콘' : 1000,
        '와쿠와쿠' : 1000,
        '월드콘' : 1000,
        '국화빵' : 1000,
        '붕어싸만코' : 1000,
        '빵또아' : 1000,
        '잇츠와플' : 1000,
        '찰떡아이스' : 1000,
        '쿠키오' : 1000,
        '구구크러스터' : 4500,
        '위즐' : 4500,
        '누가바' : 500,
        '돼지바' : 500,
        '메로나' : 500,
        '비비빅' : 500,
        '빠삐코' : 500,
        '뽕따' : 500,
        '수박바' : 500,
        '쌍쌍바' : 500,
        '옥동자' : 500,
        '죠스바' : 500,
        '주물러' : 500,
        '캔디바' : 500,
        '쿠앤크' : 500,
        '데미소다' : 1100,
        '몬스타' : 1600,
        '밀키스' : 1100,
        '스타벅스' : 1300,
        '스프라이트' : 1100,
        '쌕쌕' : 800,
        '캐나다드라이' : 1100,
        '코카콜라' : 1300,
        '환타' : 1100,
        '롤리폴리' : 2700,
        '롯데샌드' : 2700,
        '마가렛트' : 3800,
        '몽쉘' : 4000,
        '빅파이' : 3800,
        '빠다코코낫' : 2500,
        '찰떡파이' : 3800,
        '초코쿠키' : 1000,
        '쿠크다스' : 1400,
        '포키' : 1400,
    }
    
    result = [ [key, price[key], c ] for key, c in zip(keys,cnt)]
    # print(result)

    return result

