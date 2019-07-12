
import salary
import old_salary
import matplotlib.pyplot as plt

last_diff=-1
last_salary=0
x=[]
y=[]
y2=[]
y3=[]
i=0
for i in range(5000, 50000):
    diff=salary.get_salary(i) - old_salary.get_salary(i)
    x.append(i)
    y.append(diff)
    y2.append(salary.get_salary(i))
    y3.append(old_salary.get_salary(i))
    if diff-last_diff > 1:
        print("diff:{} salary:[{}-{}]".format(diff, last_salary, i))
        last_diff=diff
        last_salary=i

print("diff:{} salary:[{}-{}]".format(diff, last_salary, i))

plt.plot(x, y, 'g')
plt.plot(x, y2, 'r')
plt.plot(x, y3, 'b')
plt.show()

