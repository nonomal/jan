import { AssistantModel } from "@/_models/AssistantModel";
import { atom } from "jotai";

export const selectedModelAtom = atom<AssistantModel | undefined>(undefined);

export const activeAssistantModelAtom = atom<AssistantModel | undefined>(
  undefined
);