from gpiozero import PWMLED
from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory(host='192.168.88.226')
PWM_l = PWMLED(17, pin_factory=factory)
DIR_l = PWMLED(4, pin_factory=factory)
PWM_r = PWMLED(3, pin_factory=factory)
DIR_r = PWMLED(2, pin_factory=factory)

PWM_l.value = 0
PWM_r.value = 0
DIR_l.off()
DIR_r.off()