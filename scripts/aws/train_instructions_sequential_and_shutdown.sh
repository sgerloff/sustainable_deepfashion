#!/bin/bash
for INSTRUCTION in "$@"
do
  INSTRUCTION=$(basename ${INSTRUCTION%.json})
  echo "Instruction: instructions/$INSTRUCTION.json"
  echo "     Output: reports/$INSTRUCTION.out"

  python -m src.train_from_instruction --instruction="$INSTRUCTION.json" > reports/$INSTRUCTION.out

  echo "Done!"
  echo ""
done

sudo shutdown now -h