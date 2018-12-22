import train

# diffrent battch
for i in range(16,48,16):
    train.results(epoch=300,
                  lsize=0,
                  nodsize=100,
                  activation="sigmoid",
                  batch=i,
                  alpha=0.02)
    train.results(epoch=300,
                  lsize=1,
                  nodsize=100,
                  activation="sigmoid",
                  batch=i,
                  alpha=0.02)
    train.results(epoch=300,
                  lsize=2,
                  nodsize=100,
                  activation="sigmoid",
                  batch=i,
                  alpha=0.02)

#schıastic results

train.results(epoch=300,
              lsize=0,
              nodsize=100,
              activation="sigmoid",
              batch=1,
              alpha=0.02)
train.results(epoch=300,
              lsize=1,
              nodsize=100,
              activation="sigmoid",
              batch=1,
              alpha=0.02)
train.results(epoch=300,
              lsize=2,
              nodsize=100,
              activation="sigmoid",
              batch=1,
              alpha=0.02)
# relu results
train.results(epoch=300,
              lsize=0,
              nodsize=100,
              activation="relu",
              batch=1,
              alpha=0.02)

train.results(epoch=300,
              lsize=2,
              nodsize=100,
              activation="relu",
              batch=1,
              alpha=0.02)
# alpha resullts
train.results(epoch=300,
              lsize=0,
              nodsize=100,
              activation="sigmoid",
              batch=1,
              alpha=0.005)

train.results(epoch=300,
              lsize=2,
              nodsize=100,
              activation="sigmoid",
              batch=1,
              alpha=0.005)


