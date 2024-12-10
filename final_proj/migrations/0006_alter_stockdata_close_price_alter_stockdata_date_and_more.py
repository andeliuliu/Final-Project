# Generated by Django 5.1.4 on 2024-12-10 06:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('final_proj', '0005_stockdata_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockdata',
            name='close_price',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='date',
            field=models.DateField(blank=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='dividends',
            field=models.FloatField(blank=True, default=0),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='high_price',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='low_price',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='open_price',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='stock_splits',
            field=models.FloatField(blank=True, default=0),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='upside_downside',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='volume',
            field=models.BigIntegerField(blank=True),
        ),
    ]
