#!/bin/sh

git config --global user.email "vd.naz@yandex.ru"
git config --global user.name "Vadim Nazarov"
git config credential.helper store

while [ true ]
do
    git status
    git add experiments/*
    git status
    git commit -m"update"
    git push
    echo "sleeping...\n" 
    sleep 1200
done