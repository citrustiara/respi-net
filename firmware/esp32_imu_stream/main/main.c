#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define I2C_MASTER_SCL_IO 22      /*!< GPIO number used for I2C master clock */
#define I2C_MASTER_SDA_IO 21      /*!< GPIO number used for I2C master data  */
#define I2C_MASTER_NUM 0          /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 400000 /*!< I2C master clock frequency */
#define I2C_MASTER_TX_BUF_DISABLE 0 /*!< I2C master doesn't need buffer */
#define I2C_MASTER_RX_BUF_DISABLE 0 /*!< I2C master doesn't need buffer */
#define I2C_MASTER_TIMEOUT_MS 100

#define LSM6DS3_ADDR 0x6B /*!< Sensor address */
#define LSM6DS3_WHO_AM_I_REG 0x0F
#define LSM6DS3_CTRL1_XL 0x10
#define LSM6DS3_CTRL2_G  0x11
#define LSM6DS3_CTRL3_C  0x12
#define LSM6DS3_STATUS_REG 0x1E
#define LSM6DS3_OUTX_L_G 0x22

#define LSM6DS3_ODR_104_HZ 0x40
#define LSM6DS3_BDU_IF_INC 0x44

static const char *TAG = "IMU_STREAM";

/**
 * @brief i2c master initialization
 */
static esp_err_t i2c_master_init(void) {
  int i2c_master_port = I2C_MASTER_NUM;

  i2c_config_t conf = {
      .mode = I2C_MODE_MASTER,
      .sda_io_num = I2C_MASTER_SDA_IO,
      .scl_io_num = I2C_MASTER_SCL_IO,
      .sda_pullup_en = GPIO_PULLUP_ENABLE,
      .scl_pullup_en = GPIO_PULLUP_ENABLE,
      .master.clk_speed = I2C_MASTER_FREQ_HZ,
  };

  i2c_param_config(i2c_master_port, &conf);

  return i2c_driver_install(i2c_master_port, conf.mode,
                            I2C_MASTER_RX_BUF_DISABLE,
                            I2C_MASTER_TX_BUF_DISABLE, 0);
}

static esp_err_t lsm6ds3_register_read(uint8_t reg_addr, uint8_t *data,
                                       size_t len) {
  return i2c_master_write_read_device(
      I2C_MASTER_NUM, LSM6DS3_ADDR, &reg_addr, 1, data, len,
      I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

static esp_err_t lsm6ds3_register_write_byte(uint8_t reg_addr, uint8_t data) {
  uint8_t write_buf[2] = {reg_addr, data};
  return i2c_master_write_to_device(I2C_MASTER_NUM, LSM6DS3_ADDR, write_buf,
                                    sizeof(write_buf),
                                    I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

void app_main(void) {
  ESP_ERROR_CHECK(i2c_master_init());
  ESP_LOGI(TAG, "I2C initialized successfully");

  uint8_t who_am_i;
  ESP_ERROR_CHECK(lsm6ds3_register_read(LSM6DS3_WHO_AM_I_REG, &who_am_i, 1));
  ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", who_am_i);

  // CTRL1_XL: 104 Hz ODR, +/-2 g. CTRL2_G: 104 Hz ODR, +/-245 dps.
  ESP_ERROR_CHECK(
      lsm6ds3_register_write_byte(LSM6DS3_CTRL1_XL, LSM6DS3_ODR_104_HZ));
  ESP_ERROR_CHECK(
      lsm6ds3_register_write_byte(LSM6DS3_CTRL2_G, LSM6DS3_ODR_104_HZ));
  // Block Data Update keeps a complete sample stable during the 12-byte read;
  // IF_INC enables sequential register addressing.
  ESP_ERROR_CHECK(
      lsm6ds3_register_write_byte(LSM6DS3_CTRL3_C, LSM6DS3_BDU_IF_INC));
  ESP_LOGI(TAG, "LSM6DS3 configured for 104 Hz");

  uint8_t data[12];
  int16_t ax, ay, az, gx, gy, gz;
  uint32_t sample_index = 0;

  ESP_LOGI(TAG, "Starting 6-axis stream...");

  while (1) {
    uint8_t status;
    if (lsm6ds3_register_read(LSM6DS3_STATUS_REG, &status, 1) != ESP_OK) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }

    if ((status & 0x03) == 0x03) { // Both XL and G data ready
      if (lsm6ds3_register_read(LSM6DS3_OUTX_L_G, data, sizeof(data)) ==
          ESP_OK) {
        const int64_t device_time_us = esp_timer_get_time();

        gx = (int16_t)((data[1] << 8) | data[0]);
        gy = (int16_t)((data[3] << 8) | data[2]);
        gz = (int16_t)((data[5] << 8) | data[4]);
        ax = (int16_t)((data[7] << 8) | data[6]);
        ay = (int16_t)((data[9] << 8) | data[8]);
        az = (int16_t)((data[11] << 8) | data[10]);

        // Print: sample_index,device_time_us,ax,ay,az,gx,gy,gz
        // XL scale: 2G (0.061 mg/LSB)
        // G scale: 245 dps (8.75 mdps/LSB)
        printf("%" PRIu32 ",%" PRId64 ",%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n",
               sample_index, device_time_us,
               (float)ax * 0.061 / 1000.0,
               (float)ay * 0.061 / 1000.0,
               (float)az * 0.061 / 1000.0,
               (float)gx * 8.75 / 1000.0,
               (float)gy * 8.75 / 1000.0,
               (float)gz * 8.75 / 1000.0);
        sample_index++;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}
