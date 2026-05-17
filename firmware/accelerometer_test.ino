#include <Wire.h>
#include "SparkFunLSM6DS3.h"

#define I2C_SDA 21
#define I2C_SCL 22

LSM6DS3 myIMU(I2C_MODE, 0x6B); 
float baseZ = 0;

void setup() {
  Serial.begin(115200); 
  Wire.begin(I2C_SDA, I2C_SCL);
  
  if (myIMU.begin() != 0) while (1);

  // Kalibracja tła (200 próbek dla większej precyzji)
  for(int i=0; i<200; i++) {
    baseZ += myIMU.readFloatAccelZ();
    delay(2);
  }
  baseZ /= 200.0;
}

void loop() {
  float sumAz = 0;
  float sumAy = 0;
  int samples = 10; // Uśredniamy 10 odczytów

  for(int i=0; i < samples; i++) {
    sumAz += myIMU.readFloatAccelZ();
    sumAy += myIMU.readFloatAccelY();
    delay(4); // 10 próbek * 4ms = ok. 40ms na cykl (25Hz)
  }

  float avgAz = sumAz / samples;
  float avgAy = sumAy / samples;

  float displacement = avgAz - baseZ;
  float angle = atan2(avgAy, avgAz) * 180.0 / PI;

  Serial.print(millis());
  Serial.print(",");
  Serial.print(displacement, 4);
  Serial.print(",");
  Serial.println(angle, 2);
}