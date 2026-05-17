#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_timer.h"

const static char *TAG = "RADAR_ADC";

/*---------------------------------------------------------------
        ADC General Macros
---------------------------------------------------------------*/
// ADC1 Channel 5 is GPIO 33 on ESP32
#define EXAMPLE_ADC1_CHAN5          ADC_CHANNEL_5
// ADC_ATTEN_DB_12 (approx 0-3.1V on ESP32)
#define EXAMPLE_ADC_ATTEN           ADC_ATTEN_DB_12

// Oversampling config
#define OVERSAMPLING_COUNT          64

void app_main(void)
{
    //-------------ADC1 Init---------------//
    adc_oneshot_unit_handle_t adc1_handle;
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    //-------------ADC1 Config---------------//
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = EXAMPLE_ADC_ATTEN,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, EXAMPLE_ADC1_CHAN5, &config));

    //-------------ADC1 Calibration Init---------------//
    adc_cali_handle_t adc1_cali_handle = NULL;
    bool do_calibration1 = false;
    
    // Line Fitting calibration scheme
    adc_cali_line_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = EXAMPLE_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    esp_err_t ret = adc_cali_create_scheme_line_fitting(&cali_config, &adc1_cali_handle);
    if (ret == ESP_OK) {
        do_calibration1 = true;
        ESP_LOGI(TAG, "Calibration Success: Line Fitting");
    } else {
        ESP_LOGW(TAG, "Calibration Failed, using rough conversion");
    }

    ESP_LOGI(TAG, "Starting Radar ADC Sampling (64x Oversampling) on GPIO 33...");

    // Data transmission rate: 500 Hz (2ms period)
    const int transmission_period_ms = 2;
    TickType_t xLastWakeTime = xTaskGetTickCount();

    while (1) {
        uint32_t adc_sum = 0;
        
        // Oversampling loop
        for (int i = 0; i < OVERSAMPLING_COUNT; i++) {
            int raw;
            ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, EXAMPLE_ADC1_CHAN5, &raw));
            adc_sum += raw;
        }
        
        int adc_avg = adc_sum / OVERSAMPLING_COUNT;
        int voltage_mv_avg;
        
        if (do_calibration1) {
            ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_handle, adc_avg, &voltage_mv_avg));
        } else {
            voltage_mv_avg = (adc_avg * 3100) / 4095;
        }

        int64_t timestamp = esp_timer_get_time() / 1000;
        printf("%lld,%d,%d\n", timestamp, adc_avg, voltage_mv_avg);

        vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(transmission_period_ms));
    }
}
