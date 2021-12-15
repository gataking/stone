from django import template
register = template.Library()

def comma_three(result):
    result = str(result)
    if len(result) > 3:
        temp = ""
        cnt = 0
        for i in range(len(result)-1, -1, -1):
            cnt += 1
            if cnt % 3 == 0:
                temp = "," + result[i] + temp
            else:
                temp = result[i] + temp

        if temp[0] == ",":
            temp = temp[1:]

    return temp


@register.simple_tag
def multiply(price, amount):
    # you would need to do any localization of the result here
    return price * amount


@register.simple_tag
def tot_cnt(items):
    # you would need to do any localization of the result here
    result = 0
    for item in items:
        result += item[2]
    return result


@register.simple_tag
def tot_price(items):
    # you would need to do any localization of the result here
    result = 0
    for item in items:
        result += (item[1] * item[2])

    return comma_three(result)



