.PHONY: all revision upgrade

all: revision upgrade

revision: # run with make revision message=message
	alembic -c migrations/alembic.ini revision --autogenerate -m $(message)

upgrade:
	alembic -c migrations/alembic.ini upgrade head
