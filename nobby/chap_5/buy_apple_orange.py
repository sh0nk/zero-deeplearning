from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orrange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_all_price = mul_apple_layer.forward(apple, apple_num)
orange_all_price = mul_orrange_layer.forward(orange, orange_num)
apple_and_orange_price = add_apple_orange_layer.forward(apple_all_price, orange_all_price)
price = mul_tax_layer.forward(apple_and_orange_price, tax)

# backward
d_price = 1
d_all_price, d_tax = mul_tax_layer.backward(d_price)
d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_all_price)
d_orange, d_orange_num = mul_orrange_layer.backward(d_orange_price)
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)

print("price:", int(price))
print("d_apple:", d_apple)
print("d_apple_num:", int(d_apple_num))
print("d_orange:", d_orange)
print("d_orange_num:", int(d_orange_num))
print("d_tax:", d_tax)
