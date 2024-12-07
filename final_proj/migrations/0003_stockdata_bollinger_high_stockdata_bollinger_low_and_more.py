# Generated by Django 5.1.4 on 2024-12-06 21:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('final_proj', '0002_stockdata_upside_downside'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockdata',
            name='bollinger_high',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='bollinger_low',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='ema_12',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='ema_26',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='ema_9',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='macd',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='macd_signal',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='moving_average_10',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='moving_average_50',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='rsi',
            field=models.FloatField(blank=True, null=True),
        ),
    ]