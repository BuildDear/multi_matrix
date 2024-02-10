import asyncio
import time


async def fun1(x):
    print(x**2)
    await asyncio.sleep(3)
    print('fun1 завершена')


async def fun2(x):
    print(x**0.5)
    await asyncio.sleep(3)
    print('fun2 завершена')


async def main():
    await fun1(4)
    await fun2(4)


print(time.strftime('%X'))

asyncio.run(main())

print(time.strftime('%X'))