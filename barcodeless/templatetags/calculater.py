from django import template
register = template.Library()


# 3자리 마다 , 를 찍는 함수인데 사용 하지 않음
# def comma_three(result):
#     result = str(result)
#     if len(result) > 3:
#         temp = ""
#         cnt = 0
#         for i in range(len(result)-1, -1, -1):
#             cnt += 1
#             if cnt % 3 == 0:
#                 temp = "," + result[i] + temp
#             else:
#                 temp = result[i] + temp

#         if temp[0] == ",":
#             temp = temp[1:]

#     return temp

# 제품 출력창에서 받아온 제품갯수와 가격을 곱하여 계산하는 함수
@register.simple_tag
def multiply(price, amount):
    # you would need to do any localization of the result here
    return price * amount

# 제품 출력창의 상품갯수를 계산하는함수
@register.simple_tag
def tot_cnt(items):
    # you would need to do any localization of the result here
    result = 0
    for item in items:
        result += item[2]
    return result

# 제품 출력창의 합계금액을 계산하는함수
@register.simple_tag
def tot_price(items):
    # you would need to do any localization of the result here
    result = 0
    for item in items:
        result += (item[1] * item[2])

    return result
    # return comma_three(result)



