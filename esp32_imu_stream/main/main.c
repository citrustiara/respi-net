#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
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
#define LSM6DS3_STATUS_REG 0x1E
#define LSM6DS3_OUTX_L_G 0x22
#define LSM6DS3_OUTX_L_XL 0x28

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

    // LSM6DS3 init
    // CTRL1_XL: 0x70 = 1.66kHz ODR (XL), 2G scale
    ESP_ERROR_CHECK(lsm6ds3_register_write_byte(LSM6DS3_CTRL1_XL, 0x70));
    // CTRL2_G: 0x70 = 1.66kHz ODR (G), 250 dps
    ESP_ERROR_CHECK(lsm6ds3_register_write_byte(LSM6DS3_CTRL2_G, 0x70));
    ESP_LOGI(TAG, "LSM6DS3 configured for 1.66kHz");

  uint8_t data_xl[6];
  uint8_t data_g[6];
  int16_t ax, ay, az, gx, gy, gz;

  ESP_LOGI(TAG, "Starting 6-axis stream...");

  while (1) {
    uint8_t status;
    lsm6ds3_register_read(LSM6DS3_STATUS_REG, &status, 1);

    if ((status & 0x03) == 0x03) { // Both XL and G data ready
      if (lsm6ds3_register_read(LSM6DS3_OUTX_L_XL, data_xl, 6) == ESP_OK &&
          lsm6ds3_register_read(LSM6DS3_OUTX_L_G, data_g, 6) == ESP_OK) {
        
        ax = (int16_t)((data_xl[1] << 8) | data_xl[0]);
        ay = (int16_t)((data_xl[3] << 8) | data_xl[2]);
        az = (int16_t)((data_xl[5] << 8) | data_xl[4]);

        gx = (int16_t)((data_g[1] << 8) | data_g[0]);
        gy = (int16_t)((data_g[3] << 8) | data_g[2]);
        gz = (int16_t)((data_g[5] << 8) | data_g[4]);

        // Print: ax,ay,az,gx,gy,gz
        // XL scale: 2G (0.061 mg/LSB)
        // G scale: 250 dps (8.75 mdps/LSB)
        printf("%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n", 
               (float)ax * 0.061 / 1000.0,
               (float)ay * 0.061 / 1000.0, 
               (float)az * 0.061 / 1000.0,
               (float)gx * 8.75 / 1000.0,
               (float)gy * 8.75 / 1000.0,
               (float)gz * 8.75 / 1000.0);
      }
    }
  }
}
